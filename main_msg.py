from function_tas_amp.tas_amp import *
from function_cbba.cbba import *
from function_cbba.bwcbba import *
from function_gurobi import Gurobi_solve
from plot_result import *
from Prob import Prob
import matplotlib

matplotlib.use('Qt5Agg')

if True:
    #### 최근에 msg 그림 뽑았던 코드 250210
    prob = Prob(5, 100, 0, 2, var=0.1, discount=0.0)  # CBBA 를 위해 lambda 값 입력
    NN = 500
    tic = time.time()
    amp_cr = AMP_CR(prob, build_type=0, conf_res_type=0, refine_type=0, buff_size=2, discount=0.0, N_iter=30,
                    return_type=False)

    result = amp_cr.solve()
    toc = time.time()
    print(toc - tic)
    nn = 0
    for r in result.route:
        print(len(r), r)
        nn += len(r)
    print(nn)
    plot_result(result)
    plt.show()