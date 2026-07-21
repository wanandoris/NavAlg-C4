# 附录：项目本地git-commit记录

| 序号 | 提交哈希 | 提交时间 | 提交人 | 提交说明 |
| --- | --- | --- | --- | --- |
| 1 | ec41ba7 | 2026-06-24 09:12:24 +0800 | wanandoris | 优化结构 |
| 2 | 7c6e584 | 2026-06-24 07:48:44 +0800 | wanandoris | 新增工具与分析文档 |
| 3 | 9982301 | 2026-06-18 02:55:02 +0800 | wanandoris | README |
| 4 | 71e8cb3 | 2026-06-18 02:54:24 +0800 | wanandoris | 优化结构并新添README |
| 5 | 793dfc3 | 2026-06-17 23:24:48 +0800 | wanandoris | PPO超参及奖励优化 |
| 6 | 5762a3b | 2026-06-17 20:21:10 +0800 | wanandoris | 增加PID控制器优化输出量 |
| 7 | 53115b1 | 2026-06-16 19:39:31 +0800 | wanandoris | 修改更严厉的奖励函数 |
| 8 | 197608c | 2026-06-16 19:17:14 +0800 | wanandoris | Merge pull request #1 from Qi-huanye/main |
| 9 | 7060925 | 2026-06-16 18:29:15 +0800 | Qi-huanye | fix: correct APF debug obstacle angle logging |
| 10 | e604773 | 2026-06-16 17:02:37 +0800 | Qi-huanye | Merge remote-tracking branch 'upsream/main' |
| 11 | cbf6ba3 | 2026-06-16 15:14:40 +0800 | wanandoris | Merge branch 'main' of https://github.com/wanandoris/NavAlg-C4 |
| 12 | c9c8075 | 2026-06-16 15:13:40 +0800 | wanandoris | 修复雷达数据错误，新增可视化程序 |
| 13 | 5ad4ef3 | 2026-06-15 20:45:04 +0800 | wanandoris | 更新GAE优势 |
| 14 | 61494da | 2026-06-15 20:15:18 +0800 | Qi-huanye | 合并apf和目标过滤 |
| 15 | 5eb983f | 2026-06-15 19:22:25 +0800 | wanandoris | 增加日志绘画器 |
| 16 | 0185bee | 2026-06-15 19:15:34 +0800 | Qi-huanye | fix:通过每轮前停止3s来防止首帧失败跳轮，附加调试log模式 |
| 17 | f09ff4d | 2026-06-15 15:04:46 +0800 | CyberKanjousen | 更新 .gitignore ，修复 nav_2.py 中 compute_reward_breakdown 方法重复传入形参问题 |
| 18 | f04a6c6 | 2026-06-15 13:30:52 +0800 | wanandoris | 修复合并bug |
| 19 | d49a2f3 | 2026-06-15 13:30:35 +0800 | wanandoris | Merge branch 'main' of https://github.com/wanandoris/NavAlg-C4 |
| 20 | 4515424 | 2026-06-15 12:35:26 +0800 | wanandoris | 目标点或误识别为障碍物做优化 |
| 21 | 0b15d3d | 2026-06-15 11:08:36 +0800 | Qi-huanye | feat: 添加tesorboard可视化 |
| 22 | d61a756 | 2026-06-15 10:40:46 +0800 | Qi-huanye | chore:整理 |
| 23 | 1035a1a | 2026-06-14 15:19:44 +0800 | CyberKanjousen | 将时间奖励权重机制加到了 reward.py 中 |
| 24 | af83877 | 2026-06-14 14:45:40 +0800 | CyberKanjousen | - reward.py:5：奖励函数改为“距离进展 + 连续避障惩罚 + 航向辅助 + 速度辅助 + 每步小惩罚 + 终局奖励”。 - nav_2.py:337：调用 compute_reward(...) 时传入了 prev_distance=current_distance，让距离奖励基于 prev_distance - current_distance。 障碍物惩罚也从硬阈值改成了连续惩罚，越靠近障碍物惩罚越大。速度奖励只在方向基本正确且前方安全时鼓励前进，否则惩罚速度动作。每步额外加了 -0.01，减少拖延和绕圈。 |
| 25 | 19c1231 | 2026-06-14 13:51:50 +0800 | CyberKanjousen | 修改了一下 nav_2.py 时间奖励权重的计算的范围（现在是 [0.5, 1]） |
| 26 | 27c599c | 2026-06-14 13:47:32 +0800 | CyberKanjousen | 修改了一下 nav_1.py 时间奖励权重的计算的范围（现在是 [0.5, 1]） |
| 27 | add0a4c | 2026-06-14 13:25:19 +0800 | CyberKanjousen | 修改了一下时间奖励的计算的范围（现在是 [0, 1]），并从线形变化改成了非线性变化 |
| 28 | 4414990 | 2026-06-12 21:42:06 +0800 | wanandoris | 更新APF奖励 |
| 29 | 2cbff73 | 2026-06-12 19:11:28 +0800 | CyberKanjousen | 新增了 .gitignore |
| 30 | a4ff5f0 | 2026-06-12 19:11:00 +0800 | CyberKanjousen | 新增了时间奖励对 REWARD_ARRIVE_BONUS 的权重计算。 |
| 31 | 38f9839 | 2026-06-12 19:08:26 +0800 | CyberKanjousen | Merge remote-tracking branch 'origin/main' |
| 32 | 14fbd72 | 2026-06-12 19:05:49 +0800 | CyberKanjousen | 新增了时间奖励对 REWARD_ARRIVE_BONUS 的权重计算。 |
| 33 | 219cbb8 | 2026-06-11 19:39:52 +0800 | wanandoris | 修复部分bug |
| 34 | 88d34b7 | 2026-06-11 14:46:12 +0800 | wanandoris | 修改为二维连续动作空间 |
| 35 | a069093 | 2026-06-05 20:15:39 +0800 | wanandoris | 优化奖励与动作 |
| 36 | 7267e3d | 2026-06-05 18:19:43 +0800 | wanandoris | 增加对距离和方位的正确感知，对目标点的正确加载 |
| 37 | bf2fe70 | 2026-06-05 13:54:01 +0800 | wanandoris | 增加连续动作空间 |
| 38 | 38c1c42 | 2026-06-03 19:59:24 +0800 | wanandoris | 增加船对目标点角度的感知和main的调用 |
| 39 | 0acbbc3 | 2026-06-03 19:27:35 +0800 | wanandoris | 初始化 |
