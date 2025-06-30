import math
import torch
from tqdm import tqdm


class HOTSolver:
    def __init__(self, 
                b, 
                c, 
                m,
                n,
                cMax=1,
                max_iters=1e6, 
                tolerance=1e-6,
                check_freq=100, 
                sigma=None, 
                adjust_sigma=True,
                logging=False,
                dtype=None,
                device=None
                ):
        
        if dtype is None:
            self.dtype = torch.float64
        else:
            self.dtype = dtype
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        """_summary_

        Args:
            b : LP problem, Ax=b
            c : LP problem, <c, x>
            m : 2D Grid rows
            n : 2D Grid cols
            max_iters : maximum running iteration. Defaults to 1e6.
            tolerance : KKT stopping threshold. Defaults to 1e-6.
            check_freq : KKT check & info print frequency. Defaults to 100.
            sigma : initial sigma value. Defaults to None (built-in initialization).
            adjust_sigma : whether restart. Defaults to True.
            log : whether return log file.
        """
        # params = N in the paper
        self.params = m * n * n + n * m * m
        # since we find the vector basic operation gradually becomes a bottleneck as its length increases
        # we adopt a heuristic way to split it into small blocks according to formula:
        # blocks = \frac{size(vec)}{L1_cache} [in MB]
        self.blocks = math.ceil(self.params * 8 / 1024**2 / 16)
        self.blocks = 1 if self.blocks < 1 else self.blocks
        self.pad = self.params // self.blocks
        
        self.max_iters = max_iters
        self.check_freq = check_freq
        self.sigma = sigma
        self.adjust_sig = adjust_sigma
        self.logging = logging

        # iteration & restart count
        self.iter_count = 0
        self.inner_count = 0
        self.num_restart = 0
        
        self.c_Max = cMax

        self.errors = torch.zeros(3, dtype=self.dtype, device=self.device)
        self.zero = torch.tensor(0, dtype=self.dtype, device=self.device)
        self.dual_feasible = self.zero.clone()

        self.tolerance = torch.tensor(1e-4, dtype=self.dtype, device=self.device) if tolerance is None \
            else torch.tensor(tolerance, dtype=self.dtype, device=self.device)
            
        self.b = torch.tensor(b, dtype=self.dtype, device=self.device)
        self.c = torch.tensor(c, dtype=self.dtype, device=self.device)
        self.x = torch.zeros(self.params, dtype=self.dtype, device=self.device)  # primal problem variable
        self.x_bar = torch.zeros(self.params, dtype=self.dtype, device=self.device)
        self.x0 = self.x.clone()

        self.y = torch.zeros_like(self.b, dtype=self.dtype, device=self.device)  # dual problem variable
        self.y_bar = self.y.clone()
        self.Ax_bar = self.y.clone()

        self.z = torch.zeros_like(self.c, dtype=self.dtype, device=self.device)
        self.z_bar = torch.zeros_like(self.c, dtype=self.dtype, device=self.device)
        self.z0 = self.z.clone()
        
        # For solving linear system in proposition 4.
        self.w = (1 / m) - (1 - 1 / n) / (m + 1)
        self.Onem = -torch.ones(m - 1, dtype=self.dtype, device=self.device) / m
        self.D = torch.cat([self.Onem, torch.tensor([-(1 - 1 / n) / (m + 1)], dtype=self.dtype, device=self.device)])
        self.DRowVec = self.D / self.w
        self.reciprocal = 1 / m + 1 / n

        # For efficient indexing
        self.m2n = m * m * n
        self.mn = m * n
        self.n2 = n * n
        self.m_n = m + n
        self.m = m
        self.n = n
        
        self.onesn = torch.ones(n, dtype=self.dtype, device=self.device)
        self.onesm = torch.ones(m, dtype=self.dtype, device=self.device)
        
        self.kkt_base = None
        self.kkt_res_old = torch.tensor(torch.inf, dtype=self.dtype, device=self.device)
        
        self.log = {'kkt_error':[], 'pr_obj':[], 'dual_obj':[], 'gap':[]}

    def normalize(self):
        self.b_norm = torch.linalg.vector_norm(self.b, ord=2)
        self.c_norm = torch.linalg.vector_norm(self.c, ord=2)
        self.b_scale = self.b_norm + 1
        self.c_scale = self.c_norm + 1
        torch.div(self.b, self.b_scale, out=self.b)
        torch.div(self.c, self.c_scale, out=self.c)
        if self.sigma is None:
            self.sigma = torch.linalg.vector_norm(self.b, 2) / torch.linalg.vector_norm(self.c, 2) / math.sqrt(max(self.m, self.n))
        self.ATy_c = -self.c.clone()

    def KKT(self):
        # While Ax = b is automatically satisfied during the update process, we do not include it here
        
        torch.add(self.ATy_c, self.z_bar, out=self.ATy_c)
        torch.linalg.vector_norm(self.ATy_c, ord=2, out=self.dual_feasible)
        self.errors[1] = self.dual_feasible
        
        if self.dual_feasible <= self.tolerance:
            self.x_norm = torch.linalg.vector_norm(self.x_bar, ord=2)
            self.x_real_norm = self.x_norm * self.b_scale
            self.z_real_norm = torch.linalg.vector_norm(self.z_bar, ord=2) * self.c_scale
            self.comple = torch.minimum(self.x_bar * self.b_scale, self.z_bar * self.c_scale)
            self.complementarity_slackness = torch.linalg.vector_norm(self.comple, ord=2) / (1 + self.x_real_norm + self.z_real_norm)
            self.errors[2] = self.complementarity_slackness
        
        self.kkt_res = torch.max(self.errors)
        if self.kkt_base is None:
            self.kkt_base = self.kkt_res.clone()
            
    def compute_feaserr(self):
        torch.multiply(self.x_bar, self.b_scale, out=self.ATy_c)
        self.orgb = self.b * self.b_scale
        self.Ax()       
        primal_equality = torch.linalg.vector_norm(self.Ax_bar - self.orgb, ord=2) / self.b_scale
        primal_inequality = torch.linalg.vector_norm(torch.minimum(self.ATy_c, self.zero, out=self.ATy_c), ord=2) / (1 + torch.linalg.vector_norm(self.ATy_c, ord=2))
        self.feaserr = torch.maximum(primal_equality, primal_inequality)
        if self.logging:
            if self.device == 'cuda':
                self.log['pr_feaserr'] = self.feaserr.cpu().item()
            else:
                self.log['pr_feaserr'] = self.feaserr.item()
        
            
    def Ax(self):
        x1_r = self.ATy_c[:self.m2n]
        x1_r = x1_r.reshape(self.mn, self.m)
        torch.sum(x1_r, dim=1, out=self.Ax_bar[:self.mn])
        x2_r = self.ATy_c[self.m2n:]
        x2_r = x2_r.reshape(self.n, self.mn)
        torch.subtract(self.Ax_bar[:self.mn], torch.sum(x2_r, dim=0), out=self.Ax_bar[:self.mn])
        x1_r = x1_r.reshape(self.n, self.m, self.m)
        torch.sum(x1_r, dim=1, out=self.Ax_bar[self.mn : 2 * self.mn])
        x2_r = x2_r.reshape(self.n, self.n, self.m)
        torch.sum(x2_r, dim=1, out=self.Ax_bar[2 * self.mn : ])

    def ATy(self):
        suby = self.y_bar[:self.mn]
        torch.kron(suby, self.onesm, out=self.ATy_c[:self.m2n])
        torch.kron(self.onesn, -suby, out=self.ATy_c[self.m2n:])
        suby2 = self.y_bar[self.mn : 2 * self.mn]
        y_r2 = torch.reshape(suby2, (self.n, self.m))
        torch.add(self.ATy_c[:self.m2n], torch.kron(self.onesm, y_r2).flatten(), out=self.ATy_c[:self.m2n])
        suby3 = self.y_bar[2 * self.mn:]
        y_r3 = torch.reshape(suby3, (self.n, self.m))
        torch.add(self.ATy_c[self.m2n:], torch.kron(self.onesn, y_r3).flatten(), out=self.ATy_c[self.m2n:])

    def solveybar(self):
        R1 = self.y_bar[:self.mn]
        R2 = self.y_bar[self.mn : 2 * self.mn]
        R3 = self.y_bar[2 * self.mn:]
        R1_p = torch.reshape(R1, (self.n, self.m))
        R2_p = torch.reshape(R2, (self.n, self.m))
        R3_p = torch.reshape(R3, (self.n, self.m))
        sR_1 = torch.sum(R1_p, dim=1, keepdim=True) / self.n
        sR_2 = torch.sum(R2_p, dim=1, keepdim=True) * self.reciprocal
        s3_p = torch.sum(R3_p, dim=0, keepdim=True) / self.n
        temp = torch.sum(R3) / self.n2
        y_1 = (R1_p + sR_1 - sR_2 + s3_p + temp) / self.m_n
        Shaty_1 = torch.sum(y_1, dim=0)
        hatWShaty_1 = torch.multiply(self.D, Shaty_1) - self.D * self.DRowVec.dot(Shaty_1)
        torch.subtract(y_1, hatWShaty_1, out=y_1)
        torch.subtract(y_1, torch.sum(hatWShaty_1) / self.n, out=y_1)
        sy_1 = torch.sum(y_1, 1, keepdim=True)
        Sy_1 = torch.sum(y_1, 0, keepdim=True)
        y_2 = (R2_p - sy_1) / self.m
        y_3 = (R3_p + Sy_1) / self.n
        torch.concat([y_1.flatten(), y_2.flatten(), y_3.flatten()], out=self.y_bar)

    def update(self):
        if self.blocks != 1:
            for i in range(self.blocks):
                start, end = i * self.pad, min((i + 1) * self.pad, self.params)
                torch.div(self.x[start : end], self.sigma, out=self.ATy_c[start : end])
                torch.add(self.ATy_c[start : end], self.z[start : end], out=self.ATy_c[start : end])
                torch.subtract(self.ATy_c[start : end], self.c[start : end], out=self.ATy_c[start : end])
        else:
            torch.div(self.x, self.sigma, out=self.ATy_c)
            torch.add(self.ATy_c, self.z, out=self.ATy_c)
            torch.subtract(self.ATy_c, self.c, out=self.ATy_c)
            
        self.Ax()
        torch.div(self.b, self.sigma, out=self.y_bar)
        torch.subtract(self.y_bar, self.Ax_bar, out=self.y_bar)
        self.solveybar()
        self.ATy()
        
        if self.blocks != 1:
            for i in range(self.blocks):
                start, end = i * self.pad, min((i + 1) * self.pad, self.params)
                torch.subtract(self.ATy_c[start : end], self.c[start : end], out=self.ATy_c[start : end])
                torch.add(self.ATy_c[start : end], self.z[start : end], out=self.x_bar[start : end])
                torch.mul(self.x_bar[start : end], self.sigma, out=self.x_bar[start : end])
                torch.add(self.x_bar[start : end], self.x[start : end], out=self.x_bar[start : end])
                torch.mul(self.ATy_c[start : end], self.sigma, out=self.z_bar[start : end])
                torch.add(self.z_bar[start : end], self.x_bar[start : end], out=self.z_bar[start : end])   
                torch.maximum(-self.z_bar[start : end], self.zero, out=self.z_bar[start : end])
                torch.div(self.z_bar[start : end], self.sigma, out=self.z_bar[start : end])
                self.accstep(start, end)
        else:
            torch.subtract(self.ATy_c, self.c, out=self.ATy_c)
            torch.add(self.ATy_c, self.z, out=self.x_bar)
            torch.mul(self.x_bar, self.sigma, out=self.x_bar)
            torch.add(self.x_bar, self.x, out=self.x_bar)
            torch.mul(self.ATy_c, self.sigma, out=self.z_bar)
            torch.add(self.z_bar, self.x_bar, out=self.z_bar)   
            torch.maximum(-self.z_bar, self.zero, out=self.z_bar)
            torch.div(self.z_bar, self.sigma, out=self.z_bar)
            self.acceleration_step()
        
        self.iter_count += 1
        self.inner_count += 1
        
    def accstep(self, lower, upper):
        self.z1 = self.z[lower:upper]
        self.x1 = self.x[lower:upper]
        torch.mul(self.z1, -1, out=self.z1)
        torch.add(self.z1, self.z_bar[lower:upper] * 2, out=self.z1)
        torch.mul(self.x1, -1, out=self.x1)
        torch.add(self.x1, self.x_bar[lower:upper] * 2, out=self.x1)
        multiplier = (self.inner_count + 1) / (self.inner_count + 2)
        torch.mul(self.z1, multiplier, out=self.z1)
        torch.add(self.z1, self.z0[lower:upper] / (self.inner_count + 2), out=self.z1)
        torch.mul(self.x1, multiplier, out=self.x1)
        torch.add(self.x1, self.x0[lower:upper] / (self.inner_count + 2), out=self.x1)
        
    def acceleration_step(self):
        torch.mul(self.z, -1, out=self.z)
        torch.add(self.z, 2 * self.z_bar, out=self.z)
        torch.mul(self.x, -1, out=self.x)
        torch.add(self.x, 2 * self.x_bar, out=self.x)
        multiplier = (self.inner_count + 1) / (self.inner_count + 2)
        torch.mul(self.z, multiplier, out=self.z)
        torch.add(self.z, self.z0 / (self.inner_count + 2), out=self.z)
        torch.mul(self.x, multiplier, out=self.x)
        torch.add(self.x, self.x0 / (self.inner_count + 2), out=self.x)

    def adjust_sigma(self):
        if self.kkt_res < 0.3 * self.kkt_base or self.inner_count / self.iter_count >= 0.5 or \
                (1.2 * self.kkt_res_old < self.kkt_res < 0.9 * self.kkt_base):
                    
            torch.subtract(self.x_bar, self.x0, out=self.x0)
            torch.subtract(self.z_bar, self.z0, out=self.z0)
            delta_x = torch.linalg.vector_norm(self.x0, ord=2)
            delta_z = torch.linalg.vector_norm(self.z0, ord=2)
            temp = delta_x * (torch.linalg.vector_norm(self.x_bar) + 1) / (delta_z * (
                        torch.linalg.vector_norm(self.z_bar) + 1))
            self.sigma = torch.sqrt(temp)
            self.x0.copy_(self.x_bar)
            self.z0.copy_(self.z_bar)
            self.num_restart += 1
            self.inner_count = 0
            self.x.copy_(self.x_bar)
            self.z.copy_(self.z_bar)
            self.kkt_base.copy_(self.kkt_res)
            
    def display_info(self):
        self.prime_optimal = self.c.dot(self.x_bar) * self.bc_scale
        self.dual_optimal = self.b.dot(self.y_bar) * self.bc_scale
        self.gap = torch.abs(self.dual_optimal - self.prime_optimal) / (
                1 + torch.abs(self.dual_optimal) + torch.abs(self.prime_optimal))
            
        print("Iteration: {} / {} ||  KKT_ERROR: {} || Dual Feasibility: {} \
            || Sigma: {} || Prime-Dual Gap: {}".format(self.iter_count, self.max_iters,
                                            self.kkt_res,self.dual_feasible, self.sigma, self.gap))

    def optimize(self):
        self.normalize()
        self.bc_scale = self.b_scale * self.c_scale * self.c_Max
        print("Initial Checking...")
        self.KKT()
        self.display_info()
        print("Start optimizing, timer is initialized successfully.")
        self.flag = 1 if self.kkt_res > self.tolerance else 0
        while self.flag and self.iter_count < self.max_iters:
            self.update()
            if self.iter_count % self.check_freq == 0:
                self.KKT()
                if self.kkt_res <= self.tolerance:
                    self.flag = 0
                if self.adjust_sig:
                    self.adjust_sigma()
                self.kkt_res_old.copy_(self.kkt_res)
                self.display_info()
                
        if self.logging:
            self.log['iter'] = self.iter_count
            if self.device == 'cuda':
                self.log['kkt_error'] = self.kkt_res.cpu().item()
                self.log['pr_obj'] = self.prime_optimal.cpu().item()
                self.log['dual_obj'] = self.dual_optimal.cpu().item()
                self.log['gap'] = self.gap.cpu().item()
            else:
                self.log['kkt_error'] = self.kkt_res.item()
                self.log['pr_obj'] = self.prime_optimal.item()
                self.log['dual_obj'] = self.dual_optimal.item()
                self.log['gap'] = self.gap.item()
            return self.log
        
        
    def recover_transport_plan(self):
        torch.mul(self.x, self.b_scale, out=self.x)
        torch.maximum(self.x, self.zero, out=self.x)
        to_mid = self.x[:self.m2n].reshape((self.n, self.m, self.m)).permute(2, 1, 0)
        to_end = self.x[self.m2n:].reshape((self.n, self.n, self.m)).permute(2, 1, 0)
        
        to_mid = torch.tensor(to_mid, dtype=self.dtype, device=self.device)
        to_end = torch.tensor(to_end, dtype=self.dtype, device=self.device)
        solution = torch.zeros((self.m, self.m, self.n, self.n), dtype=self.dtype, device=self.device)
        
        # Vectorized operations
        for i in tqdm(range(self.m)):        
            for l in range(self.n):
                torch.minimum(to_mid[i, :, :], to_end[:, :, l], out=solution[i, :, :, l])
                torch.subtract(to_mid[i, :, :], solution[i, :, :, l], out=to_mid[i, :, :])
                torch.subtract(to_end[:, :, l], solution[i, :, :, l], out=to_end[:, :, l])
        
        solution = solution.permute(0, 2, 1, 3)
        solution = solution.permute(1, 0, 3, 2).reshape(self.mn, self.mn)
        
        return solution
    
    
    def recover_transport_map_gpu_sparse(self):
        self.x = self.x * self.b_scale
        to_mid = self.x[:self.m2n].reshape((self.n, self.m, self.m)).permute(2, 1, 0)
        to_end = self.x[self.m2n:].reshape((self.n, self.n, self.m)).permute(2, 1, 0)
        
        # Initialize solution as a sparse tensor
        indices = torch.empty((4, 0), dtype=torch.long, device=self.device)
        values = torch.empty((0,), dtype=self.dtype, device=self.device)
        solution = torch.sparse_coo_tensor(indices, values, (self.m, self.m, self.n, self.n), dtype=self.dtype, device=self.device)
        
        
        # Vectorized operations
        for i in tqdm(range(self.m)):        
            for l in range(self.n):
                min_values = torch.minimum(to_mid[i, :, :], to_end[:, :, l])
                non_zero_indices = min_values.nonzero(as_tuple=True)
                non_zero_values = min_values[non_zero_indices]
            
                # Update sparse solution tensor
                indices = torch.cat((indices, torch.stack((torch.full_like(non_zero_indices[0], i), non_zero_indices[0], non_zero_indices[1], torch.full_like(non_zero_indices[0], l)), dim=0)), dim=1)
                values = torch.cat((values, non_zero_values), dim=0)
                
                # Update to_mid and to_end
                to_mid[i, :, :].sub_(min_values)
                to_end[:, :, l].sub_(min_values)
    
        # Create the final sparse solution tensor
        solution = torch.sparse_coo_tensor(indices, values, (self.m, self.m, self.n, self.n), dtype=torch.float64, device=self.device)

        
        return solution
        
        