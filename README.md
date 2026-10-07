# 乙未战争：作战、信息、代理治理与长期制度演化模型

本仓库整理乙未战争及日据台湾早期占领、长期治理的仿真代码与历史结果快照。
模型用于历史机制分析，不把反事实频率解释为客观概率，也不以拟合单一伤亡数字代替
地理、兵力、制度、信息与代理网络机制。

## 三个主模型

- `simulate_yiwei_occupation.py`：地形、交通、控制、抵抗和殖民财政成本的长期状态模型。
- `simulate_yiwei_information_asymmetry.py`：马关条约后登陆时间与地点不确定、信息分层、
  领导逃亡/死亡等突变窗口模型。
- `simulate_yiwei_multilayer_network.py`：地理隔离、信息扩散、异质代理、隐藏忠诚、
  网络连坐、抵抗衰减、制度切换和长期认同变化的多层网络模型。

## 目录

- `src/`：可执行模型代码。
- `artifacts/`：CSV/JSON 仿真快照、图像和 Markdown 建模说明。

`artifacts/` 还包含战后台湾对日本依附的政治经济延伸，用于连接殖民制度、戒严、
外援依赖、身份认同与网络路径依赖；它不与 1895 年战役阶段混为同一时间尺度。

## 运行

```powershell
python -m pip install -r requirements.txt
python src/simulate_yiwei_occupation.py
python src/simulate_yiwei_information_asymmetry.py
python src/simulate_yiwei_multilayer_network.py
```

脚本把新结果写入仓库根目录的 `outputs/`。历史快照位于 `artifacts/`，用于校验趋势，
不应被当作新运行结果或精确历史观测。

## 识别边界

战役日期、登陆地点、地理走廊、主要人物与制度阶段来自公开史料；未直接观测的网络边、
忠诚状态、传播率、压制率与身份转移率是结构先验。模型应同时报告参数敏感性、消融结果
和史实走向对照，不能仅保留符合预期的一组输出。

