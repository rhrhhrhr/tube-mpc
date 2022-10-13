import numpy as np
from scipy.optimize import linprog
from scipy import linalg as lin
from math import *
from matplotlib import pyplot as plt
import casadi as ca


class Minkowski:
    def __init__(self, remove, poly):
        self.remove = remove
        self.poly = poly

    def poly_plus(self, a_1: np.matrix, b_1: np.matrix, a_2: np.matrix, b_2: np.matrix):
        if (a_1 == 0).all() and (b_1 == 0).all():
            return a_2, b_2
        elif (a_2 == 0).all() and (b_2 == 0).all():
            return a_1, b_1
        else:
            h_1 = np.mat(np.zeros((1, b_1.shape[1])))
            h_2 = np.mat(np.zeros((1, b_2.shape[1])))

            for i in range(0, a_1.shape[0]):
                h_1[0, i] = self.poly.support_fun(a_1[i, :], a_2, b_2)

            for j in range(0, a_2.shape[0]):
                h_2[0, j] = self.poly.support_fun(a_2[j, :], a_1, b_1)

            new_a = np.mat(np.vstack((a_1, a_2)))
            new_b = np.mat(np.hstack((b_1 + h_1, b_2 + h_2)))

            new_a, new_b = self.remove.collinear(new_a, new_b)
            if new_a.shape[0] > 1:
                new_a, new_b = self.remove.redundant_term(new_a, new_b)  # 去除冗余项
            # 两个多边形U = (A1,b1)，V = (A2,b2)的闵可夫斯基和是{f1_ix <= b1_i + hV(f1_i),f2_ix <= b2_i + hU(f2_i)}
            # h()是支撑函数
            return new_a, new_b
    # 两个凸多边形的闵可夫斯基和

    def poly_minus(self, a_1: np.matrix, b_1: np.matrix, a_2: np.matrix, b_2: np.matrix):
        if (a_2 == 0).all() and (b_2 == 0).all():
            return a_1, b_1
        else:
            h_1 = np.mat(np.zeros((1, b_1.shape[1])))

            for i in range(0, a_1.shape[0]):
                h_1[0, i] = self.poly.support_fun(a_1[i, :], a_2, b_2)

            new_a = a_1
            new_b = b_1 - h_1
            new_a, new_b = self.remove.collinear(new_a, new_b)  # 去除共线项
            if new_a.shape[0] > 1:
                new_a, new_b = self.remove.redundant_term(new_a, new_b)  # 去除冗余项
            # 两个多边形U = (A1,b1)，V = (A2,b2)的闵可夫斯基和是{f1_ix <= b1_i - hV(f1_i)}
            # h()是支撑函数，计算完后还需去除冗余项
            return new_a, new_b
    # 两个凸多边形的闵可夫斯基差

    @staticmethod
    def point_poly_plus(point: np.matrix, a: np.matrix, b: np.matrix):
        h_point = []

        for i in range(0, a.shape[0]):
            h_point.append(float(a[i, :] * point))

        new_a = a
        new_b = b + np.mat(h_point)

        return new_a, new_b
    # 点和凸多边形的闵可夫斯基和


class Remove:
    def __init__(self, poly):
        self.poly = poly

    @staticmethod
    def all_zero(a: np.matrix, b: np.matrix):
        c = np.hstack((a, b.T))
        delete_line = []

        for i in range(0, c.shape[0]):
            if (c[i, :-1] == 0.).all():
                delete_line.append(i)

        c = np.delete(c, delete_line, 0)

        new_a = np.mat(np.delete(c, -1, 1))
        new_b = np.mat(c[:, -1].T)

        return new_a, new_b
    # 去除系数全为零的行

    @staticmethod
    def collinear(a: np.matrix, b: np.matrix):
        c = np.hstack((a, b.T))
        delete_line = []

        for i in range(0, c.shape[0]):
            for j in range(i + 1, c.shape[0]):
                test_mat = np.vstack((c[i, :], c[j, :]))
                if np.linalg.matrix_rank(test_mat) < 2:
                    delete_line.append(j)

        c = np.delete(c, delete_line, 0)

        new_a = np.mat(np.delete(c, -1, 1))
        new_b = np.mat(c[:, -1].T)

        return new_a, new_b
    # 去除一个线性不等式矩阵(A,b)中共线的边

    def redundant_term(self, a: np.matrix, b: np.matrix):
        delete_line = []
        for i in range(0, a.shape[0]):
            c = a[i, :]
            bounds = [(None, None)] * c.shape[1]
            res = linprog(-c, A_ub=a, b_ub=b, bounds=bounds, method='revised simplex')
            if (-res.fun) + 0.0001 < float(b[0, i]):  # 由于计算时会有误差，为了防止误判加上0.0001
                delete_line.append(i)

        new_a = np.mat(np.delete(a, delete_line, 0))
        new_b = np.mat(np.delete(b, delete_line, 1))
        # 这里使用的方法是挑出A中某一行fi，计算fi*x在约束(A,b)上的最大值，如果这个值比fi对应的bi小，说明这项冗余，删除
        # 下面这些是用于去除刚好通过顶点的线，它们满足上面的条件，但只与区域有一个交点，便是顶点
        # 下面写的这个方法只能用于判断二维
        if a.shape[1] == 2:
            new_a, new_b = self.poly.edges_sort(new_a, new_b)  # 对边进行排序是为了让顶点顺序对应边的顺序
            ver = self.poly.vertex_cal(new_a, new_b)
            c = np.hstack((new_a, new_b.T))
            delete_line = []

            for i in range(0, ver.shape[0] - 1):
                if ((ver[i, :] - ver[i + 1, :]) < 0.0001).all() and ((ver[i, :] - ver[i + 1, :]) > -0.0001).all():
                    delete_line.append(i + 1)

            if ((ver[-1, :] - ver[0, :]) < 0.0001).all() and ((ver[-1, :] - ver[0, :]) > -0.0001).all():
                delete_line.append(0)

            c = np.delete(c, delete_line, 0)

            new_a = np.mat(np.delete(c, -1, 1))
            new_b = np.mat(c[:, -1].T)

        return new_a, new_b
    # 除去不等式组(A,b)中的冗余项，即找不到一个x使得小于等于号等号成立，或只有一个顶点满足等号成立


class Poly:
    @staticmethod
    def arg_cal(a: np.matrix):
        a_list = a.tolist()[0]
        cos_arg = a_list[0] / sqrt(a_list[0] ** 2 + a_list[1] ** 2)

        if float(a_list[1]) > 0:
            arg = acos(cos_arg)
        else:
            arg = -acos(cos_arg)

        return arg
    # 计算向量的辐角，用于给多边形的边排序

    def edges_sort(self, a: np.matrix, b: np.matrix):
        new_a, new_b = a.copy(), b.copy()
        if a.shape[0] < 2:
            return new_a
        else:
            last_exchange_index = 0
            sort_border = new_a.shape[0] - 1

            for i in range(0, new_a.shape[0] - 1):
                is_sorted = True
                for j in range(0, sort_border):
                    if self.arg_cal(new_a[j, :]) > self.arg_cal(new_a[j + 1, :]):
                        is_sorted = False
                        new_a[[j, j + 1], :] = new_a[[j + 1, j], :]
                        new_b[:, [j, j + 1]] = new_b[:, [j + 1, j]]
                        last_exchange_index = j  # 记住每一遍最后一个产生换序动作的序号，说明这之后的项无须排序，下一轮排序可以忽略
                sort_border = last_exchange_index
                if is_sorted:  # 如果走完一遍发现没有产生换序动作，那么说明已经排序完成，跳出循环
                    break

            return new_a, new_b
    # 冒泡排序算法，用于给多边形的边排序

    def vertex_cal(self, a: np.matrix, b: np.matrix):
        a_sort, b_sort = self.edges_sort(a, b)
        vertex = []

        for i in range(0, a_sort.shape[0] - 1):  # 按顺序两两求交点
            a_sol = a_sort[[i, i + 1], :].getA()
            b_sol = b_sort[:, [i, i + 1]].getA()[0]
            aug_mat = np.hstack((a_sort[[i, i + 1], :], b_sort[:, [i, i + 1]].T))
            if np.linalg.matrix_rank(a_sort[[i, i + 1], :]) == np.linalg.matrix_rank(aug_mat) and \
                    np.linalg.matrix_rank(aug_mat) == b_sort[:, [i, i + 1]].shape[1]:
                vertex.append(lin.solve(a_sol, b_sol))

        a_sol = a_sort[[0, -1], :].getA()  # 计算第一条边和最后一条边的交点
        b_sol = b_sort[:, [0, -1]].getA()[0]
        aug_mat = np.hstack((a_sort[[0, -1], :], b_sort[:, [0, -1]].T))
        if np.linalg.matrix_rank(a_sort[[0, -1], :]) == np.linalg.matrix_rank(aug_mat) and \
                np.linalg.matrix_rank(aug_mat) == b_sort[:, [0, -1]].shape[1]:
            vertex.append(lin.solve(a_sol, b_sol))

        return np.mat(vertex)
    # 根据已经排好的边的顺序，相邻边求交点，得出顶点，这种方法只适合凸多边形

    @staticmethod
    def support_fun(eta: np.matrix, a: np.matrix, b: np.matrix):
        bounds = [(None, None)] * eta.shape[1]
        res = linprog(-eta, A_ub=a, b_ub=b, bounds=bounds, method='revised simplex')

        return -res.fun
    # 计算凸两个多边形间的支撑函数

    def belong(self, v_a: np.matrix, v_b: np.matrix, u_a: np.matrix, u_b: np.matrix):
        res = True

        for i in range(0, u_a.shape[0]):
            res = res and (self.support_fun(u_a[i, :], v_a, v_b) <= float(u_b[0, i]))

        return res
    # 通过支撑函数判断一个凸多边形是否属于另一个凸多边形

    def plot(self, a: np.matrix, b: np.matrix, color):
        vertex = self.vertex_cal(a, b)

        vertex_list = vertex.T.tolist()

        plt.plot(vertex_list[0], vertex_list[1], color)
        plt.plot([vertex_list[0][0], vertex_list[0][-1]], [vertex_list[1][0], vertex_list[1][-1]], color)
    # 通过顶点画多边形

    @staticmethod
    def line_plot(a: np.matrix, b: np.matrix):
        for i in range(0, a.shape[0]):
            if float(a[i, 1]) == 0:
                x = []
                y = [-10 + 0.01 * i for i in range(0, int(20 / 0.01) + 1)]
                for y_item in y:
                    x.append(float(b[:, i]/a[i, 0]))
                plt.plot(x, y)
            else:
                x = [-10 + 0.01 * i for i in range(0, int(20 / 0.01) + 1)]
                y = []
                for x_item in x:
                    y.append(float(-a[i, 0]/a[i, 1]*x_item+b[:, i]/a[i, 1]))
                plt.plot(x, y)

        plt.grid(True)
        plt.show()
    # 通过画出每条边的直线来画多边形

    @staticmethod
    def is_interior(point: np.matrix, a: np.matrix, b: np.matrix):
        b_bar = (a * point).T
        res = ((b_bar - b) <= 0).all()

        return res
    # 判断点是否在多边形内


class RobustMPC:
    def __init__(self, minkowski, poly, remove):
        self.minkowski = minkowski
        self.poly = poly
        self.remove = remove

    def z_cal(self, a: np.matrix, a_w: np.matrix, b_w: np.matrix):
        alpha = 0.2
        epsilon = 0.001
        s = 0
        a_w_s = a_w
        status = False

        a_fs = np.mat([[0, 0]])
        b_fs = np.mat([[0]])

        while not status:  # 这一步是寻找给定α后，使A^s*W属于α*W的s
            s = s + 1
            a_fs, b_fs = self.minkowski.poly_plus(a_fs, b_fs, a_w_s, b_w)
            a_w_s = a_w * (np.linalg.inv(a)) ** s
            status = status or self.poly.belong(a_w_s, b_w, a_w / alpha, b_w)

        max_support = []

        for i in range(0, a_w.shape[0]):  # 这一步是为了计算确定s后，使A^s*W属于α*W的最小的α
            max_support.append(self.poly.support_fun(a_w[i, :], a_w_s, b_w) / float(b_w[0, i]))

        alpha = max(max_support)

        a_f_alpha_s = 1 / (1 - alpha) * a_fs  # 计算出F∞的逼近集合，这里下一步还可以继续逼近，但如果α够小，这个集合也足够了
        # 下面逼近的代码可以选用，可以注释掉不用，直接返回a_f_alpha_s和b_fs，逼近程度由epsilon决定
        n = 0
        ver = self.poly.vertex_cal(a_f_alpha_s, b_fs)
        max_norm = max([float(ver[i, :]*ver[i, :].T) for i in range(0, ver.shape[0])])
        a_fn = np.mat([[0, 0]])
        b_fn = np.mat([[0]])
        a_w_n = a_w
        while max_norm > epsilon:
            n = n + 1
            ver = self.poly.vertex_cal(a_f_alpha_s * (np.linalg.inv(a)) ** n, b_fs)
            max_norm = max([float(ver[i, :] * ver[i, :].T) for i in range(0, ver.shape[0])])
            a_fn, b_fn = self.minkowski.poly_plus(a_fn, b_fn, a_w_n, b_w)
            a_w_n = a_w * (np.linalg.inv(a)) ** n

        a_z, b_z = self.minkowski.poly_plus(a_f_alpha_s * (np.linalg.inv(a)) ** n, b_fs, a_fn, b_fn)

        return a_z, b_z
    # 计算正鲁棒不变集Z

    def domain_of_xf(self,
                     x_a: np.matrix,
                     x_b: np.matrix,
                     u_min,
                     u_max,
                     z_a: np.matrix,
                     z_b: np.matrix,
                     k: np.matrix
                     ):
        xf_x_a, xf_x_b = self.minkowski.poly_minus(x_a, x_b, z_a, z_b)

        bounds = [(None, None)] * k.shape[1]
        res_max = linprog(-k, A_ub=z_a, b_ub=z_b, bounds=bounds, method='revised simplex')
        res_min = linprog(k, A_ub=z_a, b_ub=z_b, bounds=bounds, method='revised simplex')
        kz_max = -res_max.fun
        kz_min = res_min.fun

        xf_u_a = np.vstack((-k, k))
        xf_u_b = np.mat([[kz_min - u_min, u_max - kz_max]])  # 由于U是线段，因此计算出KZ的最大最小值进行闵可夫斯基差

        xf_a = np.mat(np.vstack((xf_x_a, xf_u_a)))
        xf_b = np.mat(np.hstack((xf_x_b, xf_u_b)))
        # 下面的返回值偷懒了，因为知道X-Z，U-KZ的形式是x，u的取值范围，所以只返回最值，正常情况应返回不等式组
        return xf_a, xf_b, xf_x_a, xf_x_b, u_min - kz_min, u_max - kz_max
    # 计算出满足x∈X-Z，Kx∈U-KZ的不等式组

    def xf_cal(self, d_a: np.matrix, d_b: np.matrix, a: np.matrix):
        t = -1
        status = False
        a_cons, b_cons = 0, 0  # 没有意义，只是为了不出现警告

        while not status:
            t = t + 1
            a_cons = np.mat(np.zeros((d_a.shape[0] * (t + 1), d_a.shape[1])))
            b_cons = np.mat(np.zeros((1, d_a.shape[0] * (t + 1))))

            for i in range(0, d_a.shape[0]):
                for j in range(0, t + 1):
                    a_cons[i * (t + 1) + j, :] = d_a[i, :] * a ** j
                    b_cons[0, i * (t + 1) + j] = d_b[0, i]

            max_res = []

            for i in range(0, d_a.shape[0]):
                c = d_a[i, :] * (a ** (t + 1))
                bounds = [(None, None)] * d_a.shape[1]
                res = linprog(-c, A_ub=a_cons, b_ub=b_cons, bounds=bounds, method='revised simplex')
                max_res.append(-res.fun)

            status = status or ((np.mat(max_res) - d_b) <= 0).all()

        a_cons, b_cons = self.remove.collinear(a_cons, b_cons)
        a_cons, b_cons = self.remove.redundant_term(a_cons, b_cons)  # 得到结果后还需除去冗余项
        # 计算方法是增加t，直到Ot == Ot+1，于是有O∞ = Ot
        return a_cons, b_cons
    # 计算终端约束区域Xf

    @staticmethod
    def mpc_m_build(a: np.matrix, n: int) -> np.matrix:

        m = np.mat([[1, 0], [0, 1]])
        a_i = np.mat([[1, 0], [0, 1]])

        for i in range(0, n):
            a_i = a_i * a
            m = np.vstack((m, a_i))

        return m
    # 求解Xk = M*xk + C*Uk中的M矩阵，便于之后求解可行域

    @staticmethod
    def mpc_c_build(a: np.matrix, b: np.matrix, n: int) -> np.matrix or int:

        if n == 0:
            c = 0
            return c

        else:
            row_delta = b
            row_delta_d = b
            c = np.vstack((np.mat(np.zeros((b.shape[0], b.shape[1]))), b))

            for i in range(0, n - 1):
                row_delta_d = a * row_delta_d
                row_delta = np.hstack((row_delta_d, row_delta))
                c = np.hstack((c, np.mat(np.zeros((c.shape[0], b.shape[1])))))
                c = np.vstack((c, row_delta))

            return c
    # 求解Xk = M*xk + C*Uk中的C矩阵，便于之后求解可行域

    @staticmethod
    def mpc_g_h_build(x_a: np.matrix, x_b: np.matrix, xf_a: np.matrix, xf_b: np.matrix, n: int):

        g = xf_a
        h = xf_b
        row_delta = x_a

        for i in range(0, n):
            row_delta = np.hstack((row_delta, np.mat(np.zeros((row_delta.shape[0], xf_a.shape[1])))))
            g = np.hstack((np.mat(np.zeros((g.shape[0], x_a.shape[1]))), g))
            g = np.vstack((row_delta, g))
            h = np.hstack((x_b, h))

        return g, h
    # 产生关于Xk的约束不等式组

    @staticmethod
    def uk_a_b_build(u_inf, u_sup, n: int):
        uk_a = np.mat(np.zeros((2 * n, n)))
        for i in range(0, n):
            uk_a[2 * i:2 * (i + 1), i] = np.mat([[1], [-1]])

        uk_b = np.mat([[u_sup, -u_inf] * n])

        return uk_a, uk_b
    # 产生关于Uk的约束不等式组

    def feasible_set(self,
                     a: np.matrix,
                     b: np.matrix,
                     x_a: np.matrix,
                     x_b: np.matrix,
                     u_inf,
                     u_sup,
                     xf_a: np.matrix,
                     xf_b: np.matrix,
                     n: int
                     ):
        m = self.mpc_m_build(a, n)
        c = self.mpc_c_build(a, b, n)
        g, h = self.mpc_g_h_build(x_a, x_b, xf_a, xf_b, n)
        uk_a, uk_b = self.uk_a_b_build(u_inf, u_sup, n)
        a = np.vstack((np.hstack((g * m, g * c)),
                       np.hstack((np.mat(np.zeros((uk_a.shape[0], m.shape[1]))), uk_a))))
        b = np.hstack((h, uk_b))
        # 这里先求出了M，C矩阵，于是知道了Xk = M*x + C*Uk，之后将约束条件转化为G*Xk <= h，再包含A_Uk*Uk <= b_Uk
        # 于是有
        # [G*M G*C ][x ]      [h   ]
        # [        ][  ]  <=  [    ]
        # [ 0  A_Uk][Uk]      [b_Uk]
        # a, b分别代表上面两个矩阵

        for k in range(0, n):
            pos_a = []
            pos_b = []
            neg_a = []
            neg_b = []
            zero_a = []
            zero_b = []

            for i in range(0, a.shape[0]):
                if a[i, -1] > 0:
                    pos_a.append((a[i, :-1] / a[i, -1]).tolist()[0])
                    pos_b.append(b[0, i] / a[i, -1])
                elif a[i, -1] < 0:
                    neg_a.append((a[i, :-1] / (-a[i, -1])).tolist()[0])
                    neg_b.append(b[0, i] / (-a[i, -1]))
                else:
                    zero_a.append(a[i, :-1].tolist()[0])
                    zero_b.append(b[0, i])

            pos_a = np.mat(pos_a)
            pos_b = np.mat([pos_b])
            neg_a = np.mat(neg_a)
            neg_b = np.mat([neg_b])
            zero_a = np.mat(zero_a)
            zero_b = np.mat([zero_b])

            new_a = []
            new_b = []

            for i in range(0, pos_a.shape[0]):
                for j in range(0, neg_a.shape[0]):
                    new_a.append((pos_a[i, :] + neg_a[j, :]).tolist()[0])
                    new_b.append(pos_b[0, i] + neg_b[0, j])

            for i in range(0, zero_a.shape[0]):
                new_a.append(zero_a[i, :].tolist()[0])
                new_b.append(zero_b[0, i])

            new_a = np.mat(new_a)
            new_b = np.mat([new_b])

            a, b = self.remove.all_zero(new_a, new_b)
            a, b = self.remove.collinear(a, b)
            a, b = self.remove.redundant_term(a, b)
        # 以上是傅里叶-莫茨金消元法

        return a, b
    # 求解可行域，主要思路是构造满足条件的关于x，Uk的不等式组，通过Xk = M*x + C*Uk建立关系，之后利用傅里叶-莫茨金消元法
    # 将不等式组投影到x的平面上，便知道了使优化问题有解的x的取值范围

    @staticmethod
    def dm2arr(dm):
        return np.array(dm.full())
    # Ca sa di求解器类型转换

    @staticmethod
    def lqr_k_cal(a: np.matrix, b: np.matrix, q: np.matrix, r: np.matrix) -> np.matrix:
        p = np.mat(lin.solve_discrete_are(a, b, q, r))  # 求Riccati方程
        k = lin.inv(r + b.T * p * b) * b.T * p * a

        return k
    # 通过离散lqr计算出用来镇定实际状态和名义系统状态之差的矩阵K

    def main(self, x_ini: list, x_bound: dict, u_bound: dict, sys_para: dict, z: dict, xf: dict):
        # x_ini是初值，
        z_num = z['A'].shape[0]

        xf_num = xf['A'].shape[0]

        x_1 = ca.SX.sym('x_1')
        x_2 = ca.SX.sym('x_2')
        states = ca.vertcat(x_1, x_2)  # 控制器中的状态
        n_states = states.numel()  # 行数

        u = ca.SX.sym('u')
        controls = ca.vertcat(u)  # 控制器中的输入
        n_controls = controls.numel()  # 行数

        X = ca.SX.sym('X', n_states, sys_para['n'] + 1)  # 控制器中的状态变量汇总起来
        U = ca.SX.sym('U', n_controls, sys_para['n'])  # 控制器中的输入变量汇总起来
        ini = ca.SX.sym('ini', n_states)  # 储存每次求优化时的初状态

        A = ca.SX(sys_para['A'])
        B = ca.SX(sys_para['B'])  # 状态矩阵
        omega = ca.SX(np.mat([[np.random.uniform(-0.1, 0.1)], [np.random.uniform(-0.1, 0.1)]]))  # 随机的干扰

        Q = ca.SX(sys_para['Q'])
        R = ca.SX(sys_para['R'])
        P = ca.SX(sys_para['P'])  # 三个权重矩阵

        K = ca.SX(sys_para['K'])  # 状态反馈矩阵

        Z_A = ca.SX(z['A'])  # 最小的鲁棒不变集
        Xf_A = ca.SX(xf['A'])  # 终端约束

        st_fun = A @ states + B @ controls + omega
        st_fun_nom = A @ states + B @ controls

        f = ca.Function('f', [states, controls], [st_fun])  # 对应状态方程中的f()
        f_nom = ca.Function('f_nom', [states, controls], [st_fun_nom])  # 对应状态方程中的f()

        cost_fn = 0

        g = Z_A @ (ini[:n_states] - X[:, 0])  # 初值约束

        for k in range(0, sys_para['n']):
            st = X[:, k]
            con = U[:, k]
            cost_fn = cost_fn + 1 / 2 * st.T @ Q @ st + 1 / 2 * con.T @ R @ con  # 构造代价函数

            st_next = X[:, k + 1]
            st_next_val = f_nom(st, con)
            g = ca.vertcat(g, st_next - st_next_val)  # 构造约束

        cost_fn = cost_fn + 1 / 2 * X[:, sys_para['n']].T @ P @ X[:, sys_para['n']]  # 还应加上终端代价函数

        g = ca.vertcat(g, Xf_A @ X[:, sys_para['n']])  # 还有终端区域约束，输入约束加在bounds里

        opt_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))  # 将x和u均视为优化问题的变量

        nlp_prob = {
            'f': cost_fn,
            'x': opt_variables,
            'g': g,
            'p': ini
        }

        opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-12,
                'acceptable_obj_change_tol': 1e-10
            },
            'print_time': 0
        }  # 求解优化时的设置

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbx = ca.DM.zeros((n_states * (sys_para['n'] + 1) + n_controls * sys_para['n'], 1))
        ubx = ca.DM.zeros((n_states * (sys_para['n'] + 1) + n_controls * sys_para['n'], 1))

        lbx[0: n_states * (sys_para['n'] + 1): n_states] = -ca.inf  # x_1 的下界为负无穷
        lbx[1: n_states * (sys_para['n'] + 1): n_states] = -ca.inf  # x_2 的下界为负无穷

        ubx[0: n_states * (sys_para['n'] + 1): n_states] = ca.inf  # x_1 的上界为正无穷
        ubx[1: n_states * (sys_para['n'] + 1): n_states] = x_bound['sup']  # x_2 的上界

        lbx[n_states * (sys_para['n'] + 1):] = u_bound['inf']  # u 的下界
        ubx[n_states * (sys_para['n'] + 1):] = u_bound['sup']  # u 的上界

        lbg = ca.DM.zeros((n_states * sys_para['n'] + z_num + xf_num, 1))
        ubg = ca.DM.zeros((n_states * sys_para['n'] + z_num + xf_num, 1))

        for i in range(0, z_num):
            lbg[i] = -ca.inf
            ubg[i] = float(z['b'][0, i])  # x-x0始终在Z内

        for i in range(0, xf_num):
            lbg[z_num + n_states * sys_para['n'] + i] = -ca.inf
            ubg[z_num + n_states * sys_para['n'] + i] = float(xf['b'][0, i])  # x(N)在终端区域内

        args = {'lbg': lbg,
                'ubg': ubg,
                'lbx': lbx,
                'ubx': ubx
                }  # 将约束的值赋进去

        state_ini = ca.DM([x_ini[0], x_ini[1]])  # 系统初状态，要与解优化问题时的初值分清楚

        u0 = ca.DM.zeros((n_controls, sys_para['n']))
        X0 = ca.repmat(state_ini, 1, sys_para['n'] + 1)  # 求解优化过程中的初值

        cat_states = self.dm2arr(state_ini)  # 实际状态
        cat_nom_states = self.dm2arr(state_ini)  # 名义系统状态
        cat_controls = self.dm2arr(u0[:, 0])  # 实际控制量
        cat_controller_states = []  # 控制器内控制输入
        cat_controller_controls = []  # 控制器内状态

        for k in range(0, int(sys_para['T'] / sys_para['d_t'])):
            args['p'] = ca.vertcat(state_ini)  # 每一时刻都把实际状态作为优化问题的参数传回去
            args['x0'] = ca.vertcat(ca.reshape(X0, n_states * (sys_para['n'] + 1), 1), ca.reshape(u0, n_controls * sys_para['n'], 1))
            # 用上一时刻的解的变换作为下一次求优化的初值，减少求优化的计算量，防止优化问题从可行域外开始解

            sol = solver(
                x0=args['x0'],
                lbx=args['lbx'],
                ubx=args['ubx'],
                lbg=args['lbg'],
                ubg=args['ubg'],
                p=args['p']
            )

            u = ca.reshape(sol['x'][n_states * (sys_para['n'] + 1):], n_controls, sys_para['n'])  # 控制器内预测的控制输入
            X0 = ca.reshape(sol['x'][: n_states * (sys_para['n'] + 1)], n_states, sys_para['n'] + 1)  # 控制器内预测的状态
            u_star = ca.DM(u[:, 0] - K @ (state_ini - X0[:, 0]))  # tube mpc控制律

            cat_nom_states = np.dstack((cat_nom_states, self.dm2arr(X0[:, 0])))  # 记录下控制器内预测状态的第一个值
            cat_controller_states.append(self.dm2arr(X0))  # 记录下控制器内预测状态
            cat_controller_controls.append(self.dm2arr(u))  # 记录下控制器内预测的输入
            cat_controls = np.vstack((cat_controls, self.dm2arr(u_star)))  # 记录下实际控制输入

            state_ini = ca.DM.full(f(state_ini, u_star))  # 更新状态
            cat_states = np.dstack((cat_states, state_ini))  # 记录下实际状态

            u0 = ca.horzcat(
                u[:, 1:],
                ca.reshape(u[:, -1], -1, 1)
            )

            X0 = ca.horzcat(X0[:, 1:], ca.reshape(X0[:, -1], -1, 1))  # 这两步为了解优化的初值不要进入到可行域之外

        x_cat_1 = cat_states[0, :][0]
        x_cat_2 = cat_states[1, :][0]
        x_nom_cat_1 = cat_nom_states[0, :][0]
        x_nom_cat_2 = cat_nom_states[1, :][0]
        u_cat = cat_controls.T[0]  # 转换成列表

        return x_cat_1, x_cat_2, x_nom_cat_1, x_nom_cat_2, u_cat, cat_controller_states, cat_controller_controls
    # tube mpc主函数


if __name__ == "__main__":
    sim_time = 0.9
    sim_delta_t = 0.1
    sim_n = 9
    t_list = [sim_delta_t * num for num in range(0, int(sim_time / sim_delta_t))]
    A = np.mat([[1, 1], [0, 1]])
    B = np.mat([[0.5], [1]])
    Q = np.mat([[1, 0], [0, 1]])
    R = np.mat([[0.01]])
    P = np.mat([[2.0066, 0.5099], [0.5099, 1.2682]])
    x_initial = [-5, -2]
    A_W = np.mat([[1, 0], [-1, 0], [0, 1], [0, -1]])  # 干扰的约束Aw <= b中的A
    b_W = np.mat([[0.1, 0.1, 0.1, 0.1]])              # 干扰约束中的b
    A_X = np.mat([[0, 1]])  # 状态的约束Ax <= b中的A
    b_X = np.mat([[2]])     # 状态约束中的b
    U_max = 1  # U的最大值
    U_min = -1  # U的最小值

    po = Poly()
    rem = Remove(po)
    mink = Minkowski(rem, po)
    robust_mpc = RobustMPC(mink, po, rem)

    K = robust_mpc.lqr_k_cal(A, B, Q, R)
    A_K = A - B * K
    A_Z, b_Z = robust_mpc.z_cal(A_K, A_W, b_W)
    A_D, b_D, A_X_Z, b_X_Z, U_KZ_inf, U_KZ_sup = robust_mpc.domain_of_xf(A_X, b_X, U_min, U_max, A_Z, b_Z, K)
    X_sup = float(b_X_Z)
    A_Xf, b_Xf = robust_mpc.xf_cal(A_D, b_D, A_K)
    A_Xf_Z, b_Xf_Z = mink.poly_plus(A_Xf, b_Xf, A_Z, b_Z)

    X_bound = {'inf': None, 'sup': X_sup}
    U_bound = {'inf': U_KZ_inf, 'sup': U_KZ_sup}
    Z = {'A': A_Z, 'b': b_Z}
    Xf = {'A': A_Xf, 'b': b_Xf}
    system_parameter = {'A': A,
                        'B': B,
                        'Q': Q,
                        'R': R,
                        'P': P,
                        'K': K,
                        'n': sim_n,
                        'T': sim_time,
                        'd_t': sim_delta_t}

    X_1, X_2, x_nom_1, x_nom_2, u_opt, controller_x, controller_u = robust_mpc.main(x_initial, X_bound, U_bound, system_parameter, Z, Xf)

    po.plot(A_Z, b_Z, 'r')
    plt.grid(True)
    plt.show()
    po.plot(A_Xf, b_Xf, 'r')
    plt.grid(True)
    plt.show()

    # 以下被注释的代码用于观察控制器内预测的状态和输入，需要时可以查看
    # 这一段是画控制器内预测的控制输入
    '''t_controller_list = [0.1 * i for i in range(0, sim_n)]
    for item in range(0, len(controller_u)):
        plt.plot(t_controller_list, controller_u[item][0])
        plt.show()'''
    # 这一段是画控制器内预测的状态
    '''for item in range(0, len(controller_x)):
        plt.plot(controller_x[item][0], controller_x[item][1], '-.')'''
    plt.plot(X_1[:-1], X_2[:-1], '--')
    plt.plot(x_nom_1[1:], x_nom_2[1:])
    po.plot(A_Xf, b_Xf, 'b')
    po.plot(A_Xf_Z, b_Xf_Z, 'r')
    for item in range(0, len(x_nom_1) - 1):
        A_nom_Z, b_nom_Z = mink.point_poly_plus(np.mat([[x_nom_1[item + 1]], [x_nom_2[item + 1]]]), A_Z, b_Z)
        po.plot(A_nom_Z, b_nom_Z, 'g')
    plt.grid(True)
    plt.show()

    plt.plot(t_list, u_opt[1:])
    plt.show()

    A_fea, b_fea = robust_mpc.feasible_set(A, B, A_X_Z, b_X_Z, U_KZ_inf, U_KZ_sup, A_Xf, b_Xf, sim_n)
    A_fea_Z, b_fea_Z = mink.poly_plus(A_fea, b_fea, A_Z, b_Z)

    po.plot(A_fea, b_fea, 'r')
    po.plot(A_fea_Z, b_fea_Z, 'b')
    plt.grid(True)
    plt.show()