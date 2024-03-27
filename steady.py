# -*- coding: UTF-8 -*-

import math
import time
import numpy as np
from numpy.core.arrayprint import _array2string_dispatcher
from numpy.core.defchararray import array, multiply
import os
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy.core.function_base import linspace
from numpy.lib.function_base import meshgrid
from scipy import linalg

## ========================定义方程类========================== ##
#包含方程的信息
class pde:

    def exact_solution(x):
        exact=np.multiply(x,np.cos(x))
        return exact

    def exact_solution_derivative(x):
        der=np.cos(x)-np.multiply(x,np.sin(x))
        return der

    def function_a(x):
        r_a=np.exp(x)
        return r_a

    def function_f(x):
        r_f=np.multiply(-np.exp(x),np.cos(x)-2*np.sin(x)-np.multiply(x,np.cos(x))-np.multiply(x,np.sin(x)))
        return r_f

    def function_g(x):
        if x==0:
            r_g=0
        elif x==1:
            r_g=np.cos(1)
        return r_g

## 一维参考单元[-1,1]上的高斯积分点及权系数
def generate_Gauss_reference_1D(Gauss_point_number):
    if Gauss_point_number==4:
        Gauss_coefficient_reference_1D=[0.3478548451,0.3478548451,0.6521451549,0.6521451549]
        Gauss_point_reference_1D=[0.8611363116,-0.8611363116,0.3399810436,-0.3399810436]
    elif Gauss_point_number==8:
        Gauss_coefficient_reference_1D=[0.1012285363,0.1012285363,0.2223810345,0.2223810345,0.3137066459,0.3137066459,0.3626837834,0.3626837834]
        Gauss_point_reference_1D=[0.9602898565,-0.9602898565,0.7966664774,-0.7966664774,0.5255324099,-0.5255324099,0.1834346425,-0.1834346425]
    elif Gauss_point_number==2:
        Gauss_coefficient_reference_1D=[1,1]
        Gauss_point_reference_1D=[-1/np.sqrt(3),1/np.sqrt(3)]
    return Gauss_coefficient_reference_1D,Gauss_point_reference_1D

## 局部单元[a,b]上的高斯积分点及权系数
def generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound):
    Gauss_coefficient_reference_1D=np.array(Gauss_coefficient_reference_1D)
    Gauss_point_reference_1D=np.array(Gauss_point_reference_1D)
    Gauss_coefficient_local_1D=(upper_bound-lower_bound)*Gauss_coefficient_reference_1D/2
    Gauss_point_local_1D=(upper_bound-lower_bound)*Gauss_point_reference_1D/2+(upper_bound+lower_bound)/2
    return Gauss_coefficient_local_1D,Gauss_point_local_1D

## 局部基函数
def local_basis_1D(x,vertices,basis_type,basis_index,derivative_degree):
    
    result=0
    if basis_type==101:
    
        if derivative_degree==0:
        
            if basis_index==1:
                result=(vertices[1]-x)/(vertices[1]-vertices[0])
            elif basis_index==2:
                result=(x-vertices[0])/(vertices[1]-vertices[0])

        elif derivative_degree==1:
        
            if basis_index==1:
                result=1/(vertices[0]-vertices[1])
            elif basis_index==2:
                result=1/(vertices[1]-vertices[0])
    
    elif basis_type==102:
    
        bottom=np.square(vertices[0]-vertices[1])
    
        if derivative_degree==0:
        
            if basis_index==1:
                result=2*np.square(x)-(vertices[0]+3*vertices[1])*x+np.square(vertices[1])+vertices[0]*vertices[1]
            elif basis_index==2:
                result=2*np.square(x)-(3*vertices[0]+vertices[1])*x+np.square(vertices[0])+vertices[0]*vertices[1]
            elif basis_index==3:
                result=-4*np.square(x)+4*(vertices[0]+vertices[1])*x-4*vertices[0]*vertices[1]

        elif derivative_degree==1:
        
            if basis_index==1:
                result=4*x-(vertices[0]+3*vertices[1])
            elif basis_index==2:
                result=4*x-(3*vertices[0]+vertices[1])
            elif basis_index==3:
                result=-8*x+4*(vertices[0]+vertices[1])

        elif derivative_degree==2:
        
            if basis_index==1:
                result=4
            elif basis_index==2:
                result=4
            elif basis_index==3:
                result=-8
    
        result=result/bottom
    return result

## 单元上的有限元解
def FE_solution_1D(x,uh_local,vertices,basis_type,derivative_degree):
    r_solution=0
    number_of_local_basis=len(uh_local)
    for i in np.arange(number_of_local_basis).astype(int):
        r_solution=r_solution+uh_local[i]*local_basis_1D(x,vertices,basis_type,i+1,derivative_degree)
    
    return r_solution


def Gauss_quadrature_for_1D_integral_FE_solution_error(uh_local,accurate_function,vertices,Gauss_coefficient_local_1D,Gauss_point_local_1D,basis_type,derivative_degree):
    Gpn=len(Gauss_coefficient_local_1D)
    r_err=0
    for i in np.arange(Gpn).astype(int):
        r_err=r_err+Gauss_coefficient_local_1D[i]*np.square(accurate_function(Gauss_point_local_1D[i])-FE_solution_1D(Gauss_point_local_1D[i],uh_local,vertices,basis_type,derivative_degree))

    return r_err


def Gauss_quadrature_for_1D_integral_test(coefficient_function_name,Gauss_coefficient_local_1D,Gauss_point_local_1D,test_vertices,test_basis_type,test_basis_index,test_derivative_degree):
    Gpn=len(Gauss_coefficient_local_1D)

    r_test=0
    for i in np.arange(Gpn).astype(int):
        r_test=r_test+Gauss_coefficient_local_1D[i]*coefficient_function_name(Gauss_point_local_1D[i])*local_basis_1D(Gauss_point_local_1D[i],test_vertices,test_basis_type,test_basis_index,test_derivative_degree)

    return r_test


def Gauss_quadrature_for_1D_integral_trial_test(coefficient_function_name,Gauss_coefficient_local_1D,Gauss_point_local_1D,trial_vertices,trial_basis_type,trial_basis_index,trial_derivative_degree,test_vertices,test_basis_type,test_basis_index,test_derivative_degree):
    Gpn=len(Gauss_coefficient_local_1D)

    r=0
    for i in np.arange(Gpn).astype(int):
        r=r+Gauss_coefficient_local_1D[i]*coefficient_function_name(Gauss_point_local_1D[i])*local_basis_1D(Gauss_point_local_1D[i],trial_vertices,trial_basis_type,trial_basis_index,trial_derivative_degree)*local_basis_1D(Gauss_point_local_1D[i],test_vertices,test_basis_type,test_basis_index,test_derivative_degree)

    return r

## 组装刚度矩阵
def assemble_matrix_from_1D_integral(coefficient_function_name,M_partition,T_partition,T_basis_trial,T_basis_test,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,trial_basis_type,trial_derivative_degree,test_basis_type,test_derivative_degree):
    r_matrix=np.zeros((matrix_size[0],matrix_size[1]))

    for n in np.arange(number_of_elements).astype(int):
        vertices=M_partition[T_partition[:,n]-1]
        lower_bound=min(vertices[0],vertices[1])
        upper_bound=max(vertices[0],vertices[1])
        [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound)
   
        for alpha in np.arange(number_of_trial_local_basis).astype(int):
            for beta in np.arange(number_of_test_local_basis).astype(int):      
                temp=Gauss_quadrature_for_1D_integral_trial_test(coefficient_function_name,Gauss_coefficient_local_1D,Gauss_point_local_1D,vertices,trial_basis_type,alpha+1,trial_derivative_degree,vertices,test_basis_type,beta+1,test_derivative_degree) 
                r_matrix[T_basis_test[beta,n]-1,T_basis_trial[alpha,n]-1]=r_matrix[T_basis_test[beta,n]-1,T_basis_trial[alpha,n]-1]+temp

    return r_matrix

## 组装载荷向量
def assemble_vector_from_1D_integral(coefficient_function_name,M_partition,T_partition,T_basis_test,number_of_test_local_basis,number_of_elements,vector_size,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,test_basis_type,test_derivative_degree):
    r_vector=np.zeros((vector_size))

    for n in np.arange(number_of_elements).astype(int):

        vertices=M_partition[T_partition[:,n]-1]
        lower_bound=min(vertices[0],vertices[1])
        upper_bound=max(vertices[0],vertices[1])
        [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound)

        for beta in np.arange(number_of_test_local_basis).astype(int):     
            temp=Gauss_quadrature_for_1D_integral_test(coefficient_function_name,Gauss_coefficient_local_1D,Gauss_point_local_1D,vertices,test_basis_type,beta+1,test_derivative_degree)
            r_vector[T_basis_test[beta,n]-1]=r_vector[T_basis_test[beta,n]-1]+temp

    return r_vector

## 生成节点信息矩阵和单元顶点的全局指标
def generate_M_T_1D(left,right,h_partition,basis_type):
    
    h=h_partition

    if basis_type==101:

        N=(right-left)/h
        N=N.astype(int)
        M=np.zeros((N+1))
        T=np.zeros((2,N)).astype(int)

        for i in np.arange(1,N+2,1).astype(int):
            M[i-1]=left+(i-1)*h;       


        for i in np.arange(1,N+1,1).astype(int):
            T[0,i-1]=i;    
            T[1,i-1]=i+1

   
    elif basis_type==102:

        N=(right-left)/h
        N=N.astype(int)
        dh=h/2
        dN=N*2
        M=np.zeros((dN+1))
        T=np.zeros((2,dN)).astype(int)

        for i in np.arange(1,dN+2,1).astype(int):
            M[i-1]=left+(i-1)*dh       

        for i in np.arange(1,N+1,1).astype(int):
            T[0,i-1]=2*i-1   
            T[1,i-1]=2*i+1
            T[2,i-1]=2*i

    return M,T.astype(int)

## 计算节点处的最大误差
def get_maximum_error_1D(solution,N_basis,left,h_basis):
    maxerror=0
    for i in np.arange(1,N_basis+2,1).astype(int):
        temp=solution[i-1]-pde.exact_solution(left+(i-1)*h_basis)
        if np.abs(maxerror)<np.abs(temp):
            maxerror=temp

    return maxerror

## 生成边界点和边界边
def generate_boundary_nodes_1D(N_basis):
    boundary_nodes=np.zeros((3,2)).astype(int)
    boundary_nodes[0,0]=-1
    boundary_nodes[1,0]=1
    boundary_nodes[2,0]=-1
    boundary_nodes[0,1]=-1
    boundary_nodes[1,1]=N_basis+1
    boundary_nodes[2,1]=1
    return boundary_nodes

## 计算L^2/H^1误差
def FE_solution_error_1D(uh,accurate_function,left,right,h_partition,basis_type,derivative_degree,Gauss_point_number):
    N_partition=(right-left)/h_partition
    N_partition=N_partition.astype(int)
    number_of_elements=N_partition

    [M_partition,T_partition]=generate_M_T_1D(left,right,h_partition,101)

    if basis_type==102:
        [M_basis,T_basis]=generate_M_T_1D(left,right,h_partition,102)
    elif basis_type==101:
        T_basis=T_partition

    #Guass quadrature's points and weights on the refenrece interval [-1,1].
    [Gauss_coefficient_reference_1D,Gauss_point_reference_1D]=generate_Gauss_reference_1D(Gauss_point_number)

    r_FE_err=0
    #Go through all elements and accumulate the error on them.
    for n in np.arange(number_of_elements).astype(int):
    
        vertices=M_partition[T_partition[:,n]-1]
        lower_bound=min(vertices[0],vertices[1])
        upper_bound=max(vertices[0],vertices[1])
        [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound)
        uh_local=uh[T_basis[:,n]-1]
        r_FE_err=r_FE_err+Gauss_quadrature_for_1D_integral_FE_solution_error(uh_local,accurate_function,vertices,Gauss_coefficient_local_1D,Gauss_point_local_1D,basis_type,derivative_degree)
    r_FE_err=np.sqrt(r_FE_err)

    return r_FE_err

## 计算无穷范数误差
def FE_solution_error_infinity_norm_1D(uh,accurate_function,left,right,h_partition,basis_type,derivative_degree,Gauss_point_number):
    
    N_partition=(right-left)/h_partition
    N_partition=N_partition.astype(int)
    number_of_elements=N_partition

    [M_partition,T_partition]=generate_M_T_1D(left,right,h_partition,101)

    if basis_type==102:
        [M_basis,T_basis]=generate_M_T_1D(left,right,h_partition,102)
    elif basis_type==101:
        T_basis=T_partition

    #Guass quadrature's points and weights on the refenrece interval [-1,1].
    [Gauss_coefficient_reference_1D,Gauss_point_reference_1D]=generate_Gauss_reference_1D(Gauss_point_number)

    r_inf=0
    #Go through all elements and accumulate the error on them.
    for n in np.arange(number_of_elements).astype(int):
    
        vertices=M_partition[T_partition[:,n]-1]
        lower_bound=min(vertices[0],vertices[1])
        upper_bound=max(vertices[0],vertices[1])
        [Gauss_coefficient_local_1D,Gauss_point_local_1D]=generate_Gauss_local_1D(Gauss_coefficient_reference_1D,Gauss_point_reference_1D,lower_bound,upper_bound)
        uh_local=uh[T_basis[:,n]-1]
    
        temp=max(abs(accurate_function(Gauss_point_local_1D[:])-FE_solution_1D(Gauss_point_local_1D[:],uh_local,vertices,basis_type,derivative_degree)))    
        if temp>r_inf:
            r_inf=temp
    return r_inf

## 处理Dirichlet边界条件
def treat_Dirichlet_boundary_1D(Dirichlet_boundary_function_name,A,b,boundary_nodes,M_basis):
    
    row,col=boundary_nodes.shape
    nbn=col

    for k in np.arange(nbn).astype(int):

        if boundary_nodes[0,k]==-1: 
            i=boundary_nodes[1,k]
            A[i-1,:]=0
            A[i-1,i-1]=1
            b[i-1]=Dirichlet_boundary_function_name(M_basis[i-1])

    return A,b

## 处理Neumann边界条件
def treat_Neumann_boundary_1D(Neumann_boundary_function_name,b,boundary_nodes,M_basis):
    row,col=boundary_nodes.shape
    nbn=col

    for k in np.arange(nbn).astype(int):

        if boundary_nodes[0,k]==-2: 
            normal_direction=boundary_nodes[2,k]
            i=boundary_nodes[1,k]
            b[i-1]=b[i-1,0]+normal_direction*Neumann_boundary_function_name(M_basis[i-1])

    return b

## 处理Robin边界条件
def treat_Robin_boundary_1D(Neumann_boundary_function_name,Robin_boundary_function_name,A,b,boundary_nodes,M_basis):
    row,col=boundary_nodes.shape
    nbn=col

    for k in np.arange(nbn).astype(int):

        if boundary_nodes[0,k]==-3:
            normal_direction=boundary_nodes(3,k) 
            i=boundary_nodes[1,k]
            b[i-1]=b[i-1,0]+normal_direction*Neumann_boundary_function_name(M_basis[i-1])
            A[i-1,i-1]=A[i-1,i-1]+normal_direction*Robin_boundary_function_name(M_basis[i-1])

    return A,b

## 一维poisson方程求解器
def poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number):
    
    N_partition=(right-left)/h_partition
    N_partition=N_partition.astype(int)

    ## 初始化
    N_basis=N_partition
    T_basis=0
    number_of_trial_local_basis=0
    number_of_test_local_basis=0
    M_basis=0

    if basis_type==102:
        N_basis=N_partition*2
    elif basis_type==101:
        N_basis=N_partition

    [M_partition,T_partition]=generate_M_T_1D(left,right,h_partition,101)

    if basis_type==102:
        [M_basis,T_basis]=generate_M_T_1D(left,right,h_partition,102)
    elif basis_type==101:
        M_basis=M_partition
        T_basis=T_partition


    [Gauss_coefficient_reference_1D,Gauss_point_reference_1D]=generate_Gauss_reference_1D(Gauss_point_number)


    number_of_elements=N_partition
    matrix_size=np.array([N_basis+1,N_basis+1]).astype(int)
    vector_size=N_basis+1
    if basis_type==102:
        number_of_trial_local_basis=3
        number_of_test_local_basis=3
    elif basis_type==101:
        number_of_trial_local_basis=2
        number_of_test_local_basis=2

    A=assemble_matrix_from_1D_integral(pde.function_a,M_partition,T_partition,T_basis,T_basis,number_of_trial_local_basis,number_of_test_local_basis,number_of_elements,matrix_size,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,1,basis_type,1);


    b=assemble_vector_from_1D_integral(pde.function_f,M_partition,T_partition,T_basis,number_of_test_local_basis,number_of_elements,vector_size,Gauss_coefficient_reference_1D,Gauss_point_reference_1D,basis_type,0);

    boundary_nodes=generate_boundary_nodes_1D(N_basis)

    [A,b]=treat_Dirichlet_boundary_1D(pde.function_g,A,b,boundary_nodes,M_basis)


    r=linalg.solve(A, b)


    if basis_type==102:
        h_basis=h_partition/2
    elif basis_type==101:
        h_basis=h_partition

    maxerror=get_maximum_error_1D(r,N_basis,left,h_basis)
    maximum_error_at_all_nodes_of_FE=maxerror

    return r,M_basis,T_basis,A,b

## 主函数
def main():

    basis_type = 101

    left = 0
    right = 1

    Gauss_point_number = 4

    ## ======================================================= ##
    '''number=np.arange(2,7,1)
    h_partition=1/pow(2,number[1])
    [uh,Pb,Tb,A,b]=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number)
    np.savetxt("F:\\Program\\python_code\\Finite_Element_Method\\1D\\solution.txt",uh)
    np.savetxt("F:\\Program\\python_code\\Finite_Element_Method\\1D\\Pb.txt", Pb)
    np.savetxt("F:\\Program\\python_code\\Finite_Element_Method\\1D\\Tb.txt", Tb)
    np.savetxt("F:\\Program\\python_code\\Finite_Element_Method\\1D\\A.txt", A)
    np.savetxt("F:\\Program\\python_code\\Finite_Element_Method\\1D\\b.txt", b)
    infinity_error=FE_solution_error_infinity_norm_1D(uh,pde.exact_solution,left,right,h_partition,basis_type,0,Gauss_point_number)
    L2_error=FE_solution_error_1D(uh,pde.exact_solution,left,right,h_partition,basis_type,0,Gauss_point_number)
    H1_error=FE_solution_error_1D(uh,pde.exact_solution_derivative,left,right,h_partition,basis_type,1,Gauss_point_number)
    N=(1/h_partition).astype(int)
    print('h      infinity_error      L2_error       H1_error')
    print('1/%d  '%N,'  %e'%infinity_error,'   %e'%L2_error,'   %e'%H1_error)

    plt.figure()  
    u=pde.exact_solution(Pb)
    np.savetxt("F:\\Program\\python_code\\Finite_Element_Method\\1D\\exact.txt", u)
    plt.plot(Pb,u)
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('the exact solution')

    plt.figure()
    plt.plot(Pb,uh)
    plt.xlabel('x')
    plt.ylabel('$u_h$')
    plt.title('the FEM solution')

    plt.show()'''

    ## ========================计算不同步长下的数值结果=========================== ##
    print('h      infinity_error      L2_error       H1_error')
    j = 0
    number = 6 #循环次数
    N = np.zeros((number)).astype(int)
    inf_err = np.zeros((number))
    L2_err = np.zeros((number))
    H1_err = np.zeros((number))
    for i in np.arange(2,2+number,1):
        h_partition=1/pow(2,i)
        N[j] = (1/h_partition).astype(int)
        [uh,Pb,Tb,A,b]=poisson_solver_1D(left,right,h_partition,basis_type,Gauss_point_number)
        infinity_error=FE_solution_error_infinity_norm_1D(uh,pde.exact_solution,left,right,h_partition,basis_type,0,Gauss_point_number)
        inf_err[j]=infinity_error
        L2_error=FE_solution_error_1D(uh,pde.exact_solution,left,right,h_partition,basis_type,0,Gauss_point_number)
        L2_err[j]=L2_error
        H1_error=FE_solution_error_1D(uh,pde.exact_solution_derivative,left,right,h_partition,basis_type,1,Gauss_point_number)
        H1_err[j]=H1_error
        print('1/%d  '%N[j],'  %e'%infinity_error,'   %e'%L2_error,'   %e'%H1_error)
        j = j + 1
    
    R_inf_err = np.zeros((number-1))
    R_L2_err = np.zeros((number-1))
    R_H1_err = np.zeros((number-1))
    print('infinity_order  L2_order    H1_order')
    for i in np.arange(1,number,1).astype(int):
        R_inf_err[i-1]=np.log(inf_err[i-1]/inf_err[i])/np.log(2)
        R_L2_err[i-1]=np.log(L2_err[i-1]/L2_err[i])/np.log(2)
        R_H1_err[i-1]=np.log(H1_err[i-1]/H1_err[i])/np.log(2)
        print('%f     '%R_inf_err[i-1],' %f   '%R_L2_err[i-1],'%f'%R_H1_err[i-1])

    h=[1/N[j] for j in np.arange(number).astype(int)]
    plt.loglog(h,inf_err,'o-',h,L2_err,'s-',h,H1_err,'-^',[1/10,1/20],[4*pow(10,-3),1*pow(10,-3)],'--',[1/10,1/20],[24*pow(10,-3),12*pow(10,-3)])
    plt.xlabel('log(h)')
    plt.ylabel('log(error)')
    plt.title('the numerical error of FEM')
    plt.legend(['$\mathregular{L_{\infty}}$','$\mathregular{L_2}$','$\mathregular{H_1}$','$O(h^2)$','$O(h)$'])
    plt.show()

## 定义清屏函数
def clear():
    os.system('cls')

## 运行程序 
if __name__ == '__main__':
    clear() #清屏
    main()  #调用主函数
