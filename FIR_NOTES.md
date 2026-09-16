# Fir：GPU、申请节点和 fairshare

在当前 Fir 会话中查询，快照时间为 **2026-09-15 16:12 PDT**。资源状态和
fairshare 会随使用情况变化，下面也给出重新查询的命令。

`sinfo` 显示完整 GPU 为 **NVIDIA H100 80 GB**，GPU 节点配置为 4 张卡、
48 个 Slurm CPU、1152000 MB 调度内存；另有 H100 MIG 的 40/20/10 GB 配置。
本项目此前使用完整 H100，单作业申请 1 张卡、8 CPU、96 GB 内存；
后续内存申请已降至 40 GB。

当前用户 `enmingzz` 的关联信息：

| Account association | 用户 FairShare |
|---|---:|
| `def-btaati_cpu` | 0.551818 |
| `def-btaati_gpu` | 0.463219 |
| `rrg-btaati_gpu` | 0.499189 |

本实验提交时使用 `--account=rrg-btaati`，GPU 请求被站点映射到
`rrg-btaati_gpu`。批处理的完整命令见 [README](README.md#fir-batch-jobs)。
也可以从登录节点直接申请资源并启动计算节点上的交互 shell：

```bash
srun --account=rrg-btaati --nodes=1 --ntasks=1 --gpus-per-node=h100:1 \
  --cpus-per-task=8 --mem=40G --time=00:35:00 --pty bash -l
hostname
nvidia-smi
```

`srun --pty` 在分配到的计算资源上启动交互任务。
[Slurm srun 文档](https://slurm.schedmd.com/srun.html)

本项目按用户要求优先节省计算资源：默认只占用 **1 张 H100**，多个 H
串行共用一个策略服务，先做 smoke，再根据结果决定正式实验规模；任务结束
立即退出并释放资源，不保留空闲 GPU 作业。

2026-09-15 的 P=50/H=5,30 作业 `60035109` 实际运行 **19 分 10 秒**。
`seff` 记录主内存峰值 **26.34 GB**，累计 CPU 时间 40 分 14 秒，即平均
约 **2.1 核**。主内存申请缩至 **40 GB**。一次 4 核尝试（`60040618`）运行
15 分 15 秒仍未完成一个回合，已取消；尚未区分 CPU 配置和节点性能的影响。
因此保留已成功使用的 **8 核** 支撑 4 个仿真进程，避免仅按平均 CPU 用量
削减配置、反而延长 H100 占用。H=5,15,20 smoke 使用单张 H100、35 分钟上限。
资源效率应看每 GPU 分钟产出的有效回合。Slurm 的 `--time` 是运行上限，
并非必须占用的时长；此前申请 2 小时的作业也已在 19 分钟完成后结束。
[Slurm sbatch 时间限制说明](https://slurm.schedmd.com/sbatch.html#OPT_time)

8 核重试 `60041592` 也在 14 分 50 秒后取消：出现 WebSocket 心跳及握手
超时，仍无有效新回合。两次取消合计 30 分 05 秒单卡占用，因此不能将问题
归因于 CPU 数量。已加入首次推理预热、禁用服务端心跳，以及任一子任务失败
立即终止全部仿真的机制。后续先限时启动（300 秒），再限时验证一个回合
（180 秒），通过后才进行完整 smoke；失败记录保存在 `results/diagnostics/`，
不会混入成功率统计。

作业的 GPU 显存统计包含 JAX 的预分配，不能直接当成模型的最低显存需求；
未来是否使用 MIG 应根据实际工作负载验证，避免因显存不足反复重跑。
结束后用下面的命令检查用量，再调整后续请求：

```bash
seff 60035109
sacct -j 60035109 --format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU -P
```

FairShare 是调度优先级中的一个系数，值较高时该项更有利；实际开始时间还
取决于资源可用性和其他优先级项。Fair Tree 的 `LevelFS` 是同层级的
`NormShares / EffectvUsage`，大于 1 表示相对使用较少，小于 1 表示相对使用较多。
这次快照中，`rrg-btaati_gpu` 账号层面的 LevelFS 为 132.41，用户层面为
0.08517；层级排序解释了为何个人用量与最终 FairShare 不能简单一一对应。
本站查询到的使用量衰减半衰期为 7 天。
[Fair Tree 原理](https://slurm.schedmd.com/fair_tree.html)，
[sshare 字段说明](https://slurm.schedmd.com/sshare.html)。

```bash
# 限定账号，避免列出整个集群的父账号。
sshare -u "$(id -un)" -A def-btaati_cpu,def-btaati_gpu,rrg-btaati_gpu \
  --format=Account,User,RawShares,NormShares,EffectvUsage,LevelFS,FairShare

# GPU 类型与节点配置。
sinfo -h -o '%G|%c|%m' | rg 'gpu:' | sort -u

# 自己的作业和等待原因。
squeue -u "$(id -un)" -o '%.18i %.12j %.8T %.10M %R'
```
