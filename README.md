# 智慧物业多Agent协作系统

基于 LangChain + FastAPI + Chroma RAG 的多 Agent 物业管理系统，提供公告语义搜索、物业费管理、业主管理等功能。

## 项目结构

```
MyAgent/
├── src/
│   ├── agents/
│   │   ├── __init__.py          # 模块导出
│   │   ├── admin_agent.py       # 管理员 Agent（12工具：公告/物业费/业主 CRUD）
│   │   └── resident_agent.py    # 住户 Agent（RAG公告搜索 + 物业费查询）
│   ├── property_app.py          # 物业系统 FastAPI 服务端（端口 9090）
│   ├── property_db.py           # 数据层（SQLite + Chroma + Excel）
│   ├── property_agents.py       # 向后兼容重导出
│   ├── agent_builder.py         # HR 助手 Agent 工厂
│   ├── app.py                   # HR 助手 FastAPI 服务端（端口 8080）
│   ├── DeepSeek_v4_pro.py       # LLM 实例化
│   ├── tools.py                 # 基础工具（数学计算 + 时间）
│   └── env.py                   # 环境变量加载
├── ADVER/                       # 公告源文件（TXT/PDF）
├── OWNER/
│   ├── 业主信息.xlsx             # 业主数据
│   └── 物业费.xlsx               # 物业费数据
├── RAG/                         # HR 助手知识库文件
├── chroma_announcements_db/     # 公告 Chroma 向量库
├── property.db                  # SQLite 数据库
├── property_login.html          # 物业系统前端页面
└── test.html                    # HR 助手前端页面
```

## 快速启动

### 环境要求

- Python 3.10+
- 依赖包: langchain, langchain-community, fastapi, chromadb, openpyxl, sentence-transformers, python-multipart, uvicorn

### 启动物业管理系统（端口 9090）

```bash
python src/property_app.py
```

浏览器打开 `http://localhost:9090`

### 启动HR助手（端口 8080）测试连通性

```bash
python src/app.py
```

浏览器打开 `http://localhost:8080`

## 物业管理系统使用说明

### 登录账号

| 角色 | 账号 | 密码 |
|------|------|------|
| 管理员 | admin | admin123 |
| 住户 101 | 门牌号 101 | 101001 |
| 住户 102 | 门牌号 102 | 102001 |
| 住户 201 | 门牌号 201 | 201001 |
| 住户 302 | 门牌号 302 | 302001 |

### 管理员功能

登录后显示 3 个标签页的仪表盘：

#### 1. AI助手

通过自然语言对话管理物业，支持以下操作：

- **公告管理**: 查看全部、新增、修改、删除公告
- **物业费管理**: 查看全部记录、新增记录、标记已缴纳、删除记录
- **业主管理**: 查看全部业主、新增、修改、删除业主信息

#### 2. 公告文件

- **上传文件**: 支持 TXT 和 PDF 格式。TXT 文件第一行为标题，后续行为正文；PDF 文件以文件名为标题，自动提取全部文本
- **文件列表**: 展示文件名、类型、大小、上传日期、关联公告 ID
- **删除文件**: 同时删除文件和对应的数据库记录
- **重建索引**: 清空全部公告并从文件重新导入。适用于批量替换文件或修复索引

#### 3. 业主管理

- **新增业主**: 填写门牌号、姓名、电话、密码后添加
- **编辑业主**: 点击"编辑"进入行内编辑模式，修改后保存
- **删除业主**: 确认后删除

### 住户功能

登录后进入客服聊天界面，支持：

- **公告查询**: 自然语言语义搜索，如"最近有什么停水通知？"、"小区有什么活动？"
- **物业费查询**: 查询本户的物业费缴纳情况

## RAG 语义搜索

系统使用 Qwen3-Embedding-0.6B 模型（本地 CPU 运行）将公告文本转换为向量，存储在 Chroma 向量数据库中。用户用自然语言提问时，系统通过语义相似度匹配最相关的公告，无需精确关键词。

- **双写策略**: 公告的增删改操作同步写入 SQLite（结构化存储）和 Chroma（向量索引）
- **回退机制**: RAG 搜索无结果时自动回退到 SQLite 的 LIKE 模糊搜索

## 多 Agent 架构

```
住户提问"我的物业费交了吗"
    ↓
ResidentAgent 调用 query_my_payment 工具
    ↓
工具内部验证住户身份 (Excel)
    ↓
向管理员系统查询物业费数据 (Excel)
    ↓
返回原始 JSON → ResidentAgent 的 LLM 翻译为自然语言
    ↓
"您当前有1笔未缴纳的物业费：2026年下半年，金额1200元..."
```

## API 接口

### 认证

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /login | 统一登录（管理员/住户） |

### 对话

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /chat/admin | 管理员 Agent 对话 |
| POST | /chat/resident | 住户 Agent 对话 |

### 公告文件管理（管理员专用）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/adver/files | 列出 ADVER/ 文件 |
| POST | /api/adver/upload | 上传公告文件 |
| DELETE | /api/adver/files/{filename} | 删除文件及关联公告 |
| POST | /api/adver/reindex | 重建全部索引 |

### 业主管理（管理员专用）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/owners | 列出全部业主 |
| POST | /api/owners | 新增业主 |
| PUT | /api/owners/{room_number} | 修改业主信息 |
| DELETE | /api/owners/{room_number} | 删除业主 |

## 数据存储

| 数据类型 | 存储方式 | 路径 |
|----------|----------|------|
| 公告 | SQLite + Chroma 向量库 | property.db / chroma_announcements_db/ |
| 公告源文件 | TXT/PDF 文件 | ADVER/ |
| 业主信息 | Excel | OWNER/业主信息.xlsx |
| 物业费 | Excel | OWNER/物业费.xlsx |
