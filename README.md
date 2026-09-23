# 心脏病早期风险预测（Heart Disease Early-Stage Risk Prediction）

基于 UCI Heart Disease 数据集（克利夫兰子集）的机器学习项目：利用常规体检中容易获得的 13 项医学指标，预测个体是否患有心脏病。项目覆盖从数据探索、严谨预处理、多模型对比、医疗导向评估、可解释性分析到跨中心外部验证与 Web 演示应用的完整流程。

> ⚠️ 本项目仅用于教学与研究演示，不构成任何医疗建议。

---

## 1. 项目目标

- **医学目标**：在严重症状出现之前识别心脏病高风险个体，辅助医生决定是否安排进一步检查（筛查场景）。
- **技术目标**：
  - 以 Pipeline 封装预处理与模型，从机制上杜绝数据泄露；
  - 以交叉验证 + Bootstrap 置信区间进行严谨的模型比较与选择；
  - 以筛查视角优化决策阈值（敏感性优先），并通过校准曲线、决策曲线分析（DCA）评估临床可用性；
  - 以 SHAP / PDP / 置换重要性提供全局与个体层面的可解释性；
  - 通过匈牙利子库外部验证考察模型的跨中心泛化能力；
  - 以 Streamlit 将模型交付为可交互的演示应用。
- **参考基线**：Detrano 等（1989）在同一数据集上的逻辑回归判别函数准确率约 77%；Gennari 的 CLASSIT 为 78.9%。本项目模型性能已超越上述历史基线。

## 2. 数据说明

- 数据来源：UCI Machine Learning Repository, Heart Disease Data Set（1988 年 7 月由 David Aha 捐赠公开）。
- 训练数据：`data/processed.cleveland.data`（克利夫兰诊所子集，303 条 × 14 属性，缺失值以 `?` 标记）。
- 外部验证数据：`data_raw/processed.hungarian.data` 等（匈牙利、瑞士、长滩子库，缺失严重，仅用作外部测试）。
- 标签处理：原始标签 num 为 0〜4（0=无心脏病，1〜4=患病程度递增），二值化为 `target = (num > 0)`（0 类 164 条，1 类 139 条）。
- 注意：数据包内官方 WARNING 文件指出 `cleveland.data` 已损坏，本项目不使用。

## 3. 目录结构

```
heart-disease-project/
├── data/                       # 训练数据与官方属性文档
│   ├── processed.cleveland.data
│   ├── heart-disease.names
│   └── heart_raw.csv           # 原始数据副本（只读备份）
├── data_raw/                   # 外部验证子库（匈牙利/瑞士/长滩）
├── notebooks/                  # 五个分析笔记本（按流程编号）
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_interpretation.ipynb
├── figures/                    # 全部插图（报告与 PPT 共用）
├── app/                        # Streamlit 演示应用与模型文件
│   ├── app.py
│   └── heart_model.joblib
└── README.md
```

## 4. 代码功能说明（各 Notebook 做什么）

| Notebook | 功能 |
|---|---|
| `01_eda.ipynb` | 读入数据并校验（303×14，ca 缺失 4、thal 缺失 2）；标签二值化；单变量分布图、按是否患病的分组箱线图/堆叠条形图；13 个变量的 t 检验/卡方检验；相关热力图 |
| `02_preprocessing.ipynb` | 分层划分训练/测试集（8:2，random_state=42）；用 ColumnTransformer + Pipeline 封装缺失值填补、独热编码、标准化；验证管线可训练 |
| `03_modeling.ipynb` | 无脑基线与历史基线；五个经典分类器（逻辑回归/KNN/决策树/随机森林/SVM）的 5 折交叉验证对比；ROC 曲线；XGBoost/LightGBM；GridSearchCV 调参；Bootstrap 95% 置信区间；最终模型定选 |
| `04_evaluation.ipynb` | 阈值优化（敏感性 ≥ 0.90 反推阈值）；校准曲线；决策曲线分析（净获益）；漏诊/误诊病例的错误分析 |
| `05_interpretation.ipynb` | 三重特征重要性（逻辑回归系数、随机森林内置、置换重要性）；SHAP 蜂群图与单病人瀑布图；PDP 部分依赖图；匈牙利子库外部验证 |

## 5. 关键函数与管线设计

- **预处理管线 `preprocess`（ColumnTransformer）**
  - 数值列（age, trestbps, chol, thalach, oldpeak）：`SimpleImputer(strategy="median")` → `StandardScaler()`
  - 无序类别列（cp, restecg, thal）：`SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(drop="first", handle_unknown="ignore")`
  - 二值/有序列（sex, fbs, exang, slope, ca）：`SimpleImputer(strategy="most_frequent")` 后原样通过
  - 设计要点：所有统计量只在训练折内拟合，交叉验证时自动防数据泄露。
- **`decision_curve(y_true, proba, thresholds)`（04_evaluation.ipynb）**：按净获益公式 NB = TP/n − FP/n × pₜ/(1−pₜ) 计算模型策略在各阈值概率下的净获益，与"所有人干预/都不干预"两条极端策略对比。
- **Bootstrap 置信区间（03_modeling.ipynb）**：对测试集有放回重采样 2000 次，取 AUC 分布的 2.5%/97.5% 分位数作为 95% CI。
- **`app/app.py`**：加载 `heart_model.joblib`（预处理+模型一体的整条 Pipeline），接收 13 项指标输入，输出患病概率并按校准阈值（0.32，筛查取向）给出高/低风险提示。

## 6. 主要结果

- 五模型 5 折交叉验证 AUC：逻辑回归 0.899 ± 0.046（领先），随机森林 0.881，SVM 0.877，KNN 0.841，决策树 0.828。
- 最终模型测试集 AUC 约 0.95（95% CI 见 `03_modeling.ipynb`）。
- 阈值优化：默认 0.5 → 下调至约 0.32，敏感性提升至 0.90 以上（筛查场景漏诊代价远高于误诊）。
- 可解释性：thal（铊试验）、thalach（最大心率）、ca（显影血管数）、cp（胸痛类型）在三种重要性方法与 SHAP 中一致居前，与医学常识相符。
- 外部验证：匈牙利子库 AUC 降至约 0.8，体现跨中心分布偏移；缺失严重的特征（如 ca）跨中心失效，生理指标（thalach、oldpeak）相对稳健。

## 7. 如何复现

```bash
# 1. 环境（Python 3.11）
pip install pandas numpy matplotlib seaborn scikit-learn \
            xgboost lightgbm shap streamlit joblib scipy jupyter

# 2. 按编号顺序运行 notebooks/ 下的五个 Notebook

# 3. 启动演示应用
cd app
streamlit run app.py
```

## 8. 数据来源与致谢

使用本数据集的成果应注明各机构数据采集负责人：

- 匈牙利心脏病研究所（布达佩斯）：Andras Janosi 博士
- 瑞士苏黎世大学医院：William Steinbrunn 博士
- 瑞士巴塞尔大学医院：Matthias Pfisterer 博士
- 长滩退伍军人医疗中心与克利夫兰诊所：Robert Detrano 博士
- 数据捐赠者：David W. Aha（1988 年 7 月）

参考文献：

1. Janosi A, Steinbrunn W, Pfisterer M, Detrano R. Heart Disease Data Set[DB/OL]. UCI Machine Learning Repository, 1988. https://archive.ics.uci.edu/dataset/45/heart-disease
2. Detrano R, et al. International application of a new probability algorithm for the diagnosis of coronary artery disease[J]. American Journal of Cardiology, 1989, 64(5): 304-310.
3. Lundberg S M, Lee S I. A unified approach to interpreting model predictions[C]// NeurIPS, 2017: 4765-4774.
4. Vickers A J, Elkin E B. Decision curve analysis: A novel method for evaluating prediction models[J]. Medical Decision Making, 2006, 26(6): 565-574.

## 9. 版权与许可说明

- **代码**：本仓库代码采用 MIT License 发布，可自由使用、修改与分发，须保留版权声明。作者不承担因使用本代码产生的任何后果。
- **数据**：UCI Heart Disease 数据集版权归原始数据采集机构与捐赠者所有，依据 UCI Machine Learning Repository 的使用条款（CC BY 4.0）仅用于教学与研究目的；如基于本数据发表成果，请按第 8 节要求注明数据来源。
- **免责声明**：本项目为教学演示项目，模型基于上世纪 80 年代末的单中心小样本数据训练，预测结果不构成医疗建议，不可用于任何真实临床场景。
