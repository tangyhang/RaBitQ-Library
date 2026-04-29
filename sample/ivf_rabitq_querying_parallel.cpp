#include <iostream>
#include <vector>
#include <omp.h> // 引入 OpenMP 头文件

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static std::vector<size_t> get_nprobes(
    const index_type& ivf,
    const std::vector<size_t>& all_nprobes,
    data_type& query,
    gt_type& gt
);

static size_t topk = 100;
static size_t test_round = 5;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <arg1> <arg2> <arg3> <arg4>\n"
                  << "arg1: path for index \n"
                  << "arg2: path for query file, format .fvecs\n"
                  << "arg3: path for groundtruth file format .ivecs\n"
                  << "arg4: whether use high accuracy fastscan, (\"true\" or \"false\"), "
                     "true by default\n\n";
        exit(1);
    }

    char* index_file = argv[1];
    char* query_file = argv[2];
    char* gt_file = argv[3];
    bool use_hacc = true;

    if (argc > 4) {
        std::string hacc_str(argv[4]);
        if (hacc_str == "false") {
            use_hacc = false;
            std::cout << "Do not use Hacc FastScan\n";
        }
    }

    data_type query;
    gt_type gt;
    rabitqlib::load_vecs<float, data_type>(query_file, query);
    rabitqlib::load_vecs<uint32_t, gt_type>(gt_file, gt);
    size_t nq = query.rows();
    size_t total_count = nq * topk;

    index_type ivf;
    ivf.load(index_file);

    rabitqlib::StopW stopw;

    std::vector<size_t> nprobes = {20, 50, 100, 150, 200, 500};
    size_t length = nprobes.size();

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));

    // 【关键修改1】提前分配好所有查询的结果存储空间，完全避开循环内的动态分配开销
    std::vector<std::vector<PID>> all_results(nq, std::vector<PID>(topk));

    for (size_t r = 0; r < test_round; r++) {
        for (size_t l = 0; l < length; ++l) {
            size_t nprobe = nprobes[l];
            if (nprobe > ivf.num_clusters()) {
                std::cout << "nprobe " << nprobe << " is larger than number of clusters, ";
                std::cout << "will use nprobe = num_cluster (" << ivf.num_clusters() << ").\n";
            }
            
            // ---------------- 1. 计时与搜索阶段 ----------------
            stopw.reset(); 

            int thread_count = 64; // 设定为你想要的线程数

            // 只在这个循环中开启 8 个线程
            #pragma omp parallel for num_threads(thread_count)
            for (size_t i = 0; i < nq; i++) {
                // 每个线程将结果写入预先分配好的独立内存地址，不会产生数据竞争
                ivf.search(&query(i, 0), topk, nprobe, all_results[i].data(), use_hacc);
            }
            
            // 仅将纯搜索时间计入墙上时间
            float total_time = stopw.get_elapsed_micro(); 

            // ---------------- 2. 精度评估阶段 ----------------
            size_t total_correct = 0;
            
            // 精度计算同样可以使用多线程加速，通过 reduction 安全合并正确数
            #pragma omp parallel for reduction(+:total_correct)
            for (size_t i = 0; i < nq; i++) {
                for (size_t j = 0; j < topk; j++) {
                    for (size_t k = 0; k < topk; k++) {
                        if (gt(i, k) == all_results[i][j]) {
                            total_correct++;
                            break;
                        }
                    }
                }
            }

            // ---------------- 3. 指标计算阶段 ----------------
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            float recall =
                static_cast<float>(total_correct) / static_cast<float>(total_count);

            all_qps[r][l] = qps;
            all_recall[r][l] = recall;
        }
    }

    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "nprobe\tQPS\trecall" << '\n';

    for (size_t i = 0; i < length; ++i) {
        size_t nprobe = nprobes[i];
        float qps = avg_qps[i];
        float recall = avg_recall[i];

        std::cout << nprobe << '\t' << qps << '\t' << recall << '\n';
    }

    return 0;
}

static std::vector<size_t> get_nprobes(
    const index_type& ivf,
    const std::vector<size_t>& all_nprobes,
    data_type& query,
    gt_type& gt
) {
    size_t nq = query.rows();
    size_t total_count = topk * nq;
    float old_recall = 0;
    std::vector<size_t> nprobes;

    // 同样，为辅助函数预分配内存
    std::vector<std::vector<PID>> all_results(nq, std::vector<PID>(topk));

    for (auto nprobe : all_nprobes) {
        nprobes.push_back(nprobe);

        // 搜索阶段
        #pragma omp parallel for
        for (size_t i = 0; i < nq; i++) {
            ivf.search(&query(i, 0), topk, nprobe, all_results[i].data());
        }

        // 验证阶段
        size_t total_correct = 0;
        #pragma omp parallel for reduction(+:total_correct)
        for (size_t i = 0; i < nq; i++) {
            for (size_t j = 0; j < topk; j++) {
                for (size_t k = 0; k < topk; k++) {
                    if (gt(i, k) == all_results[i][j]) {
                        total_correct++;
                        break;
                    }
                }
            }
        }

        float recall = static_cast<float>(total_correct) / static_cast<float>(total_count);
        if (recall > 0.997 || recall - old_recall < 1e-5) {
            break;
        }
        std::cout << recall << '\t' << nprobe << std::endl << std::flush;
        old_recall = recall;
    }

    return nprobes;
}