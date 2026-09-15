# Fir：GPU、申请节点和 fairshare

在当前 Fir 会话中查询，快照时间为 **2026-09-15 16:12 PDT**。资源状态和
fairshare 会随使用情况变化，下面也给出重新查询的命令。

`sinfo` 显示完整 GPU 为 **NVIDIA H100 80 GB**，GPU 节点配置为 4 张卡、
48 个 Slurm CPU、1152000 MB 调度内存；另有 H100 MIG 的 40/20/10 GB 配置。
本项目实际使用完整 H100，单作业申请 1 张卡、8 CPU、96 GB 内存。

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
  --cpus-per-task=8 --mem=96G --time=03:00:00 --pty bash -l
hostname
nvidia-smi
```

`srun --pty` 在分配到的计算资源上启动交互任务。
[Slurm srun 文档](https://slurm.schedmd.com/srun.html)

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
