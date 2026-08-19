# Phoenix 正式产品离线激活

Phoenix 源码/开发版默认不锁定。正式发布包通过 `16_产品发布/PHOENIX_PRODUCT_RELEASE.json` 进入产品模式；产品模式下，GUI 与 CLI 业务功能都必须先通过离线激活。

## 授权设计

- 医院电脑无需联网。
- 首次启动显示机器码，例如 `PHX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX`。
- 授权电脑使用私钥针对该机器码签发 `PHX1...` 激活码。
- Phoenix 产品只携带 Ed25519 公钥，不携带私钥。
- 激活码可包含客户、版本、授权编号、永久/到期时间及功能集合。
- `activation.lic` 保存在项目 SSD 的 `16_产品发布/授权/`，Git 已忽略。
- 激活文件复制到另一台电脑后会因机器码不匹配而失效。

## 正式发布前一次性初始化

在仅由产品所有者控制的授权电脑上运行：

```bash
python release_license_tool.py keygen --out D:/phoenix_private_license_keys
```

生成的 `phoenix_license_private_key.pem` 永远不要提交 Git、不要放入医院产品 SSD、不要发送给客户。公钥可以进入发布包。

## 制作正式发布包

```bash
python release_license_tool.py prepare-release \
  --project-root D:/project_phoenix \
  --public-key D:/phoenix_private_license_keys/phoenix_license_public_key.pem \
  --version 1.0.0 \
  --edition Professional
```

该命令复制公钥，并写入正式产品标记。正式发布包从此强制激活。

## 给客户签发离线激活码

客户/医院首次启动后提供机器码。授权电脑执行：

```bash
python release_license_tool.py issue \
  --private-key D:/phoenix_private_license_keys/phoenix_license_private_key.pem \
  --machine-code PHX-XXXX-XXXX-XXXX-XXXX-XXXX-XXXX \
  --customer "某医院放射科" \
  --edition Professional
```

不指定 `--expires` 即永久授权；如需期限授权可指定 `--expires 2027-12-31`。

## CLI 授权管理

```bash
python app.py --machine-code --no-gui
python app.py --license-status --no-gui
python app.py --activate "PHX1...." --no-gui
```

正式 GUI 启动时如果未激活，会显示专用激活窗口；激活成功后进入工作台。

## 安全边界

当前 Python 版采用非对称签名，能够阻止简单伪造、复制激活文件和修改激活码。最终商业发布仍建议进一步打包为 EXE，并对发行包进行代码签名/完整性校验；任何纯 Python 客户端授权都不能等同于不可破解 DRM。
