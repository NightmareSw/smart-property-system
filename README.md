# 智慧物业多Agent协作系统

基于 LangChain + FastAPI + Chroma RAG 的多 Agent 物业管理系统，提供公告语义搜索、物业费管理、在线支付、报修工单、消息通知等功能。

## 项目结构

```
MyAgent/
├── src/
│   ├── agents/
│   │   ├── __init__.py          # 模块导出
│   │   ├── admin_agent.py       # 管理员 Agent（25工具）
│   │   └── resident_agent.py    # 住户 Agent（4工具）
│   ├── property_app.py          # 物业系统 FastAPI 服务端（端口 9090）
│   ├── property_db.py           # 数据层（SQLite 6表 + Chroma 向量库）
│   ├── property_agents.py       # 向后兼容重导出
│   ├── agent_builder.py         # HR 助手 Agent 工厂
│   ├── app.py                   # HR 助手 FastAPI 服务端（端口 8080）
│   ├── DeepSeek_v4_pro.py       # LLM 实例化
│   ├── tools.py                 # 基础工具（数学计算 + 时间）
│   └── env.py                   # 环境变量加载
├── ADVER/                       # 公告源文件（TXT/PDF）
├── OWNER/                       # 业主/物业费 Excel（种子数据源）
├── admin/                       # 管理员账号 Excel（种子数据源）
├── chroma_announcements_db/     # 公告 Chroma 向量库
├── property.db                  # SQLite 数据库（6张表）
├── property_login.html          # 物业系统前端页面
└── test.html                    # HR 助手前端页面
```

## 快速启动

### 环境要求

- Python 3.10+
- 依赖: langchain, langchain-community, fastapi, chromadb, openpyxl, sentence-transformers, python-multipart, uvicorn

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

### 登录账号（可在项目中的admin与OWN中添加，或通过前端demo进行注册）

| 角色 | 账号 | 密码 |
|------|------|------|
| 管理员 | admin | admin123 |
| 住户 101 | 门牌号 101 | 101001 |
| 住户 102 | 门牌号 102 | 102001 |
| 住户 201 | 门牌号 201 | 201001 |
| 住户 302 | 门牌号 302 | 302001 |

### 管理员功能（5 标签页仪表盘）

#### AI助手
通过自然语言对话管理物业：
- 公告管理：查看、搜索、AI生成草稿（确认后发布）、新增、修改、删除
- 物业费管理：查看全部、按状态筛选、按门牌号查询、新增、标记已缴纳、删除、手动切换状态(paid/unpaid)
- 业主管理：查看、新增、修改、删除（含密码管理）
- 报修工单：查看全部工单、更新状态+备注
- 管理员账号管理：新增、修改、删除管理员

#### 公告文件
- 上传 TXT/PDF 文件（TXT首行为标题，PDF以文件名作标题）
- 文件列表含关联公告ID
- 删除文件同步删除数据库记录
- 重建索引：清空全部公告并从文件重新导入

#### 业主管理
- 新增业主（门牌号、姓名、电话、密码）
- 表格展示 + 行内编辑
- 密码列支持显隐切换
- 确认后删除

#### 工单管理
- 查看全部报修工单（门牌号、标题、描述、状态）
- 点击更新按钮修改状态（待处理/处理中/已完成/已取消）+ 备注
- 更新后自动通知对应住户

#### 物业费查询
- 发布表单：选择业主(全部/指定)、金额、年份、周期(月/季/半年/全年)、备注
- 确认后批量创建缴费记录，自动通知对应业主，记录操作日志
- 按门牌号分组卡片，显示记录数、未缴总额
- 点击展开该户全部缴费明细（含备注），每条记录可**切换缴纳状态**
- 底部显示最近发布记录，支持**撤销**（删除该批次全部记录）

### 住户功能（4 标签页界面）

#### AI客服
- 公告查询：自然语言 RAG 语义搜索
- 物业费查询：查看自己的缴费情况
- 报修提交：描述问题即可提交工单

#### 公告查看
- 公告列表（按发布时间倒序）
- 点击展开公告全文

#### 物业缴费
- 缴费列表（金额、截止日期、状态）
- 未缴纳记录旁「立即支付」按钮
- 确认弹窗 → 模拟支付成功 → 自动标记已缴纳

#### 报修
- 提交表单（标题 + 详细描述）
- 我的工单列表（查看处理状态和物业回复）

### 消息通知
- 顶部铃铛图标 + 未读红点
- 点击展开通知列表，点击已读
- 公告发布、物业费发布、工单更新时自动推送

## RAG 语义搜索

系统使用 Qwen3-Embedding-0.6B 模型（本地 CPU 运行）将公告文本转换为向量，存储在 Chroma 向量数据库中。住户用自然语言提问时，系统通过语义相似度匹配最相关的公告，无需精确关键词。

- 双写策略：公告增删改同步写入 SQLite + Chroma
- 回退机制：RAG 无结果时自动回退到 SQLite LIKE 模糊搜索

## 多 Agent 架构

```
住户: "我家水管漏水，需要报修"
    ↓
ResidentAgent 调用 submit_repair 工具
    ↓
创建工单 → 通知管理员
    ↓
管理员: "查看全部工单" → AdminAgent 调用 admin_list_repairs
    ↓
管理员更新状态+备注 → 通知住户
    ↓
住户: "查看我的工单" → 显示处理进度和物业回复
```

## 数据库

| 表 | 内容 | 说明 |
|----|------|------|
| announcements | 公告 | SQLite + Chroma 双写 |
| owners | 业主信息 | 首次从 Excel 种子导入 |
| payments | 物业费记录 | 含notes备注列 |
| admins | 管理员账号 | 首次从 Excel 种子导入 |
| notifications | 消息通知 | 运行时自动生成 |
| repairs | 报修工单 | 支持状态流转 |
| payment_logs | 物业费操作日志 | 发布/撤销记录 |

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

### 管理员账号管理（管理员专用）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/admins | 列出全部管理员 |
| POST | /api/admins | 新增管理员 |
| PUT | /api/admins/{username} | 修改管理员信息 |
| DELETE | /api/admins/{username} | 删除管理员 |

### 物业费（管理员+住户）
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/payments/grouped | 管理员按门牌号分组查询 |
| POST | /api/payments/publish | 管理员发布物业费 |
| POST | /api/payments/set-status | 管理员手动切换缴纳状态 |
| GET | /api/payments/logs | 管理员查看发布操作日志 |
| POST | /api/payments/undo/{log_id} | 管理员撤销发布操作 |
| GET | /api/my-payments | 住户查看自己的缴费记录 |
| POST | /api/pay/{id} | 住户模拟支付 |

### 报修工单
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/repairs | 管理员查看全部工单 |
| POST | /api/repairs | 住户提交报修 |
| GET | /api/repairs/my | 住户查看自己的工单 |
| PUT | /api/repairs/{id} | 管理员更新工单状态 |

### 消息通知
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/notifications | 获取通知列表 |
| GET | /api/notifications/unread-count | 未读数量 |
| POST | /api/notifications/{id}/read | 标记已读 |

### 公开接口
| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/announcements | 公告列表 |
| GET | /api/get_announcement/{id} | 公告详情 |

## 数据存储

| 数据类型 | 存储方式 | 路径 |
|----------|----------|------|
| 公告 | SQLite + Chroma 向量库 | property.db / chroma_announcements_db/ |
| 公告源文件 | TXT/PDF 文件 | ADVER/ |
| 业主信息 | SQLite（Excel种子） | OWNER/业主信息.xlsx |
| 物业费 | SQLite（Excel种子） | OWNER/物业费.xlsx |
| 管理员账号 | SQLite（Excel种子） | admin/管理员.xlsx |
| 消息通知 | SQLite | property.db |
| 报修工单 | SQLite | property.db |
| 物业费操作日志 | SQLite | property.db |
