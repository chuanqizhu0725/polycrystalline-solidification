#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "mpi.h"

#define NDX 200
#define NDY 200
#define N 10
#define BEGIN 1
#define UTAG 2
#define DTAG 3
#define NONE 0
#define DONE 4
#define MASTER 0

int ndx = NDX;
int ndy = NDY;
int ndmx = NDX - 1;
int ndmy = NDY - 1;          //計算領域の一辺の差分分割数(差分ブロック数), ND-1を定義
int nm = N - 1, nmm = N - 2; //考慮する結晶方位の数、N-2（考慮する結晶方位の数－１）を定義
double PI = 3.141592;        //π、計算カウント数
double RR = 8.3145;          //ガス定数

double aij[N][N]; //勾配エネルギー係数
double wij[N][N]; //ペナルティー項の係数
double mij[N][N]; //粒界の易動度
double fij[N][N]; //粒界移動の駆動力
int phinum;

int i, j, k, l, ii, jj, kk, ll, it; //整数
int ip, im, jp, jm;                 //整数
int n1, n2, n3;                     //整数

int istep = 0;
// int n000;		//位置(i,j)において、pが０ではない方位の個数（n00>=n000）
int nstep;               //計算カウント数の最大値（計算終了カウント）
double dtime, L, dx;     // L計算領域の一辺の長さ(nm), 差分プロック１辺の長さ(m)
double M0;               //粒界の易動度
double W0;               //ペナルティー項の係数
double A0;               //勾配エネルギー係数
double F0;               //粒界移動の駆動力
double temp;             //温度
double sum1, sum2, sum3; //各種の和の作業変数
double pddtt;            //フェーズフィールドの時間変化率

double gamma0; //粒界エネルギ密度
double delta;  //粒界幅（差分ブロック数にて表現）
double mobi;   //粒界の易動度
double vm0;    //モル体積

int x11, y11, x1h[10], y1h[10]; //初期核の座標
double t, r0, r;

//******* メインプログラム ******************************************
int main(int argc, char *argv[])
{
    nstep = 1001;
    dtime = 5.0;
    temp = 1000.0;
    L = 2000.0;
    vm0 = 7.0e-6;
    delta = 7.0;
    mobi = 1.0;

    dx = L / 100.0 * 1.0e-9;             //差分プロック１辺の長さ(m)
    gamma0 = 0.5 * vm0 / RR / temp / dx; //粒界エネルギ密度（0.5J/m^2）を無次元化
    A0 = 8.0 * delta * gamma0 / PI / PI; //勾配エネルギー係数[式(4.40)]
    W0 = 4.0 * gamma0 / delta;           //ペナルティー項の係数[式(4.40)]
    M0 = mobi * PI * PI / (8.0 * delta); //粒界の易動度[式(4.40)]
    F0 = 80.0 / RR / temp;               //粒界移動の駆動力

    for (i = 0; i <= nm; i++)
    {
        for (j = 0; j <= nm; j++)
        {
            wij[i][j] = W0;
            aij[i][j] = A0;
            mij[i][j] = M0;
            fij[i][j] = 0.0;
            if ((i == 0) || (j == 0))
            {
                fij[i][j] = F0;
            }
            if (i < j)
            {
                fij[i][j] = -fij[i][j];
            }
            if (i == j)
            {
                wij[i][j] = 0.0;
                aij[i][j] = 0.0;
                mij[i][j] = 0.0;
                fij[i][j] = 0.0;
            }
        }
    }

    int taskid,
        numworkers,
        numtasks,
        rows, offset,
        dest, source,
        up, down,
        msgtype,
        rc, start, end,
        ix, iy, iz, it;

    MPI_Status status;

    // Allocate taskid to each core (cores = tasks = master + workers)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);

    numworkers = numtasks - 1;
    rows = NDX / numworkers;

    /************************* master code *******************************/
    if (taskid == MASTER)
    {
        clock_t start_t, end_t, total_t;
        start_t = clock();
        if (NDX % numworkers != 0)
        {
            MPI_Abort(MPI_COMM_WORLD, rc);
            exit(1);
        }

        double(*phi)[N][NDX][NDY] = malloc(sizeof(*phi)); //フェーズフィールド、フェーズフィールド補助配列
        double(*intphi)[NDX][NDY] = malloc(sizeof(*intphi));

        for (i = 0; i <= ndmx; i++)
        {
            for (j = 0; j <= ndmy; j++)
            {
                for (ii = 1; ii <= nm; ii++)
                {
                    (*phi)[ii][i][j] = 0.0;
                }
                (*phi)[0][i][j] = 1.0;
            }
        }

        r0 = 20.0;
        for (ii = 1; ii <= nm; ii++)
        {
            x11 = rand() % 100 * (ndx / 100);
            y11 = rand() % 100 * (ndy / 100); //初期核の位置
            for (i = 0; i <= ndmx; i++)
            {
                for (j = 0; j <= ndmy; j++)
                {
                    r = sqrt(((i - x11)) * (i - x11) + (j - y11) * (j - y11));
                    if (r <= r0)
                    {
                        (*phi)[ii][i][j] = 1.0;
                        for (k = 0; k <= nm; k++)
                        {
                            if (k != ii)
                            {
                                (*phi)[k][i][j] = 0.0;
                            }
                        }
                    } //初期核位置のフェーズフィールドを設定
                }
            }
        }

        offset = 0;
        // Send to workers
        for (i = 1; i <= numworkers; i++)
        {
            dest = i;
            if (dest == 1)
                up = NONE;
            else
                up = dest - 1;
            if (dest == numworkers)
                down = NONE;
            else
                down = dest + 1;

            MPI_Send(&offset, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(&up, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            MPI_Send(&down, 1, MPI_INT, dest, BEGIN, MPI_COMM_WORLD);
            //// send phase fields
            for (ii = 0; ii <= nm; ii++)
            {
                MPI_Send((*phi)[ii][offset], rows * NDY, MPI_DOUBLE, dest, BEGIN, MPI_COMM_WORLD);
            }

            offset = offset + rows;
        }

        // Receive from workers
        for (i = 1; i <= numworkers; i++)
        {
            source = i;
            msgtype = DONE;
            MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD,
                     &status);
            MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
            //// receive phase fields
            MPI_Recv((*intphi)[offset], rows * NDY, MPI_DOUBLE, source,
                     msgtype, MPI_COMM_WORLD, &status);
        }

        FILE *fp;

        fp = fopen("intphi.dat", "w");

        for (int i = 0; i <= ndmx; i++)
        {
            for (int j = 0; j <= ndmy; j++)
            {
                fprintf(fp, "%e\n", (*intphi)[i][j]);
            }
        }
        fclose(fp);
        end_t = clock();
        total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
        printf("Total time taken: %lu secs\n", total_t);
        printf("Exiting of the program...\n");

        MPI_Finalize();
    }

    /************************* workers code **********************************/
    if (taskid != MASTER)
    {
        double(*phi)[N][rows + 2][NDY] = malloc(sizeof(*phi));
        double(*phi2)[N][rows + 2][NDY] = malloc(sizeof(*phi2));
        int(*phiNum)[rows + 2][NDY] = malloc(sizeof(*phiNum));
        int(*phiIdx)[N + 1][rows + 2][NDY] = malloc(sizeof(*phiIdx));
        double(*intphi)[rows + 2][NDY] = malloc(sizeof(*intphi));

        // Receive from master
        source = MASTER;
        msgtype = BEGIN;
        MPI_Recv(&offset, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&rows, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&up, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        MPI_Recv(&down, 1, MPI_INT, source, msgtype, MPI_COMM_WORLD, &status);
        //// receive phase fields
        for (ii = 0; ii <= nm; ii++)
        {
            MPI_Recv((*phi)[ii][1], rows * NDY, MPI_DOUBLE, source, msgtype, MPI_COMM_WORLD, &status);
        }

    start:;

        // Communicate with neighor works before computation
        if (up != NONE)
        {
            //// send up boundaries of phase fields
            for (ii = 0; ii <= nm; ii++)
            {
                MPI_Send((*phi)[ii][1], NDY, MPI_DOUBLE, up, DTAG, MPI_COMM_WORLD);
            }

            source = up;
            msgtype = UTAG;
            //// receive up boundaries of phase fields
            for (ii = 0; ii <= nm; ii++)
            {
                MPI_Recv((*phi)[ii][0], NDY, MPI_DOUBLE, source,
                         msgtype, MPI_COMM_WORLD, &status);
            }
        }
        if (down != NONE)
        {
            //// send down boundaries of phase fields
            for (ii = 0; ii <= nm; ii++)
            {
                MPI_Send((*phi)[ii][rows], NDY, MPI_DOUBLE, down,
                         UTAG, MPI_COMM_WORLD);
            }

            source = down;
            msgtype = DTAG;
            //// receive down boundaries of phase fields
            for (ii = 0; ii <= nm; ii++)
            {
                MPI_Recv((*phi)[ii][rows + 1], NDY, MPI_DOUBLE, source, msgtype,
                         MPI_COMM_WORLD, &status);
            }
        }

        // Compute after sending and receiving data
        start = 1;
        end = rows;

        for (i = start; i <= end; i++)
        {
            for (j = 0; j <= ndmy; j++)
            {
                ip = i + 1;
                im = i - 1;
                jp = j + 1;
                jm = j - 1;
                if (up == NONE && i == 1)
                {
                    im = 1;
                }
                if (down == NONE && i == rows)
                {
                    ip = rows;
                }
                if (j == ndmy)
                {
                    jp = 0;
                }
                if (j == 0)
                {
                    jm = ndmy;
                }

                //--- 位置(i,j)およびその周囲(i±1,j±1)において、pが０ではない方位の個数---
                phinum = 0;
                for (ii = 0; ii <= nm; ii++)
                {
                    if (((*phi)[ii][i][j] > 0.0) ||
                        (((*phi)[ii][i][j] == 0.0) && ((*phi)[ii][ip][j] > 0.0) ||
                         ((*phi)[ii][im][j] > 0.0) ||
                         ((*phi)[ii][i][jp] > 0.0) ||
                         ((*phi)[ii][i][jm] > 0.0)))
                    {
                        phinum++;
                        (*phiIdx)[phinum][i][j] = ii;
                        // printf("%d  ", n00);
                    }
                }
                (*phiNum)[i][j] = phinum;
            }
        }

        // Evolution Equations
        for (i = start; i <= end; i++)
        {
            for (j = 0; j <= ndmy; j++)
            {
                ip = i + 1;
                im = i - 1;
                jp = j + 1;
                jm = j - 1;
                if (up == NONE && i == 1)
                {
                    im = 1;
                }
                if (down == NONE && i == rows)
                {
                    ip = rows;
                }
                if (j == ndmy)
                {
                    jp = 0;
                }
                if (j == 0)
                {
                    jm = ndmy;
                }

                for (n1 = 1; n1 <= (*phiNum)[i][j]; n1++)
                {
                    ii = (*phiIdx)[n1][i][j];
                    pddtt = 0.0;
                    for (n2 = 1; n2 <= (*phiNum)[i][j]; n2++)
                    {
                        jj = (*phiIdx)[n2][i][j];
                        sum1 = 0.0;
                        for (n3 = 1; n3 <= (*phiNum)[i][j]; n3++)
                        {
                            kk = (*phiIdx)[n3][i][j];
                            sum1 += 0.5 * (aij[ii][kk] - aij[jj][kk]) * ((*phi)[kk][ip][j] + (*phi)[kk][im][j] + (*phi)[kk][i][jp] + (*phi)[kk][i][jm] - 4.0 * (*phi)[kk][i][j]) + (wij[ii][kk] - wij[jj][kk]) * (*phi)[kk][i][j]; //[式(4.31)の一部]
                        }
                        pddtt += -2.0 * mij[ii][jj] / (double)((*phiNum)[i][j]) * (sum1 - 8.0 / PI * fij[ii][jj] * sqrt((*phi)[ii][i][j] * (*phi)[jj][i][j]));
                        //フェーズフィールドの発展方程式[式(4.31)]
                    }
                    (*phi2)[ii][i][j] = (*phi)[ii][i][j] + pddtt * dtime; //フェーズフィールドの時間発展（陽解法）
                    if ((*phi2)[ii][i][j] >= 1.0)
                    {
                        (*phi2)[ii][i][j] = 1.0;
                    } //フェーズフィールドの変域補正
                    if ((*phi2)[ii][i][j] <= 0.0)
                    {
                        (*phi2)[ii][i][j] = 0.0;
                    }
                }
            } // j
        }     // i

        for (k = 0; k <= nm; k++)
        {
            for (i = start; i <= end; i++)
            {
                for (j = 0; j <= ndmy; j++)
                {
                    (*phi)[k][i][j] = (*phi2)[k][i][j];
                }
            }
        }

        for (i = start; i <= end; i++)
        {
            for (j = 0; j <= ndmy; j++)
            {
                sum1 = 0.0;
                for (k = 0; k <= nm; k++)
                {
                    sum1 += (*phi)[k][i][j];
                }
                for (k = 0; k <= nm; k++)
                {
                    (*phi)[k][i][j] = (*phi)[k][i][j] / sum1;
                }
            }
        }

        istep = istep + 1;
        if (istep < nstep)
        {
            goto start;
        }
    end:;

        for (i = start; i <= end; i++)
        {
            for (j = 0; j <= ndmy; j++)
            {
                for (ii = 0; ii <= nm; ii++)
                {
                    (*intphi)[i][j] += (*phi)[ii][i][j] * (*phi)[ii][i][j];
                }
            }
        }
        // Send final result to master
        MPI_Send(&offset, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, MASTER, DONE, MPI_COMM_WORLD);
        //// send phase fields
        MPI_Send((*intphi)[1], rows * NDY, MPI_DOUBLE, MASTER, DONE,
                 MPI_COMM_WORLD);
        MPI_Finalize();
    }
    return 0;
}
