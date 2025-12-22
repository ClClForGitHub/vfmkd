# SSH 自动登录配置指南

## 问题描述
Cursor通过SSH连接时频繁要求输入密码，影响使用体验。

## 解决方案

### 方案1：使用SSH密钥对（强烈推荐）⭐

这是最安全和最方便的方法，配置后完全不需要输入密码。

#### 步骤1：检查本地是否已有SSH密钥
```bash
ls -la ~/.ssh/id_ed25519*
```
如果看到 `id_ed25519` 和 `id_ed25519.pub`，说明密钥已存在。

#### 步骤2：将公钥复制到远程服务器

**方法A：使用 ssh-copy-id（最简单）**
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub 用户名@服务器IP或域名
```
例如：
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub root@192.168.1.100
```

**方法B：手动复制**
```bash
# 1. 查看公钥内容
cat ~/.ssh/id_ed25519.pub

# 2. 复制输出的内容，然后登录到远程服务器
ssh 用户名@服务器IP

# 3. 在远程服务器上执行
mkdir -p ~/.ssh
chmod 700 ~/.ssh
echo "粘贴你的公钥内容" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

#### 步骤3：测试密钥登录
```bash
ssh 用户名@服务器IP
```
如果能直接登录（不需要密码），说明配置成功！

#### 步骤4：在SSH配置文件中添加服务器信息

编辑 `~/.ssh/config`，添加你的服务器配置：

```bash
Host myserver  # 自定义别名，可以改成你喜欢的名字
    HostName 192.168.1.100  # 你的服务器IP或域名
    User root  # 你的用户名
    Port 22  # SSH端口（默认22）
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

然后在Cursor中连接时，直接使用别名 `myserver` 即可。

---

### 方案2：使用sshpass自动输入密码（备选方案）⚠️

**注意**：这种方法不够安全，密码会以明文形式存储。

#### 步骤1：安装sshpass
```bash
# Ubuntu/Debian
sudo apt-get install sshpass

# CentOS/RHEL
sudo yum install sshpass

# 或者从源码编译
wget http://sourceforge.net/projects/sshpass/files/sshpass/1.06/sshpass-1.06.tar.gz
tar xvzf sshpass-1.06.tar.gz
cd sshpass-1.06
./configure
make
sudo make install
```

#### 步骤2：在SSH配置中使用
编辑 `~/.ssh/config`：

```bash
Host myserver-password
    HostName 你的服务器IP
    User 你的用户名
    Port 22
    ProxyCommand sshpass -p '你的密码' ssh -o StrictHostKeyChecking=no -W %h:%p %r
```

**安全警告**：密码会出现在配置文件中，任何能读取该文件的人都能看到你的密码。

---

### 方案3：使用SSH Agent（推荐用于多个服务器）

如果你有多个服务器，可以使用SSH Agent管理密钥：

```bash
# 启动SSH Agent
eval $(ssh-agent)

# 添加密钥
ssh-add ~/.ssh/id_ed25519

# 查看已添加的密钥
ssh-add -l
```

这样配置后，所有使用该密钥的服务器都可以自动登录。

---

## 在Cursor中使用

1. **安装Remote SSH扩展**（如果还没安装）
   - 打开扩展市场，搜索 "Remote - SSH"

2. **连接到服务器**
   - 按 `F1` 或 `Ctrl+Shift+P`
   - 输入 "Remote-SSH: Connect to Host"
   - 选择你配置的服务器别名（如 `myserver`）

3. **首次连接**
   - 如果使用密钥，应该可以直接连接
   - 如果仍然要求输入密码，说明密钥没有正确配置到服务器

---

## 常见问题

### Q1: 配置了密钥但仍然要求输入密码？
- 检查远程服务器的 `~/.ssh/authorized_keys` 文件权限（应该是600）
- 检查远程服务器的 `~/.ssh` 目录权限（应该是700）
- 确认公钥已正确添加到 `authorized_keys` 文件中

### Q2: 如何查看SSH连接日志？
```bash
ssh -v 用户名@服务器IP  # 显示详细信息
ssh -vv 用户名@服务器IP  # 显示更详细信息
```

### Q3: 如何在多个服务器上使用同一个密钥？
- 只需要将同一个公钥添加到所有服务器的 `~/.ssh/authorized_keys` 文件中即可

### Q4: 如何禁用密码登录（更安全）？
在远程服务器的 `/etc/ssh/sshd_config` 中设置：
```
PasswordAuthentication no
PubkeyAuthentication yes
```
然后重启SSH服务：`sudo systemctl restart sshd`

---

## 推荐做法

1. **优先使用SSH密钥**（方案1）
2. 为每个服务器创建专用的SSH配置条目
3. 定期更换密钥（如果密钥泄露）
4. 不要使用密码自动输入（方案2），除非有特殊需求

---

## 你的公钥信息

你的SSH公钥已经存在：
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC/xAAOu7bkXDnoi3L8HTNEn23shWeqd/0co0zHupMXK zouzhiyuan@h3c-R5300-G5-20251101
```

使用命令 `cat ~/.ssh/id_ed25519.pub` 可以随时查看。

