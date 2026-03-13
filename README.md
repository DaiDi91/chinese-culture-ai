初学者从零使用AI制作过程（半成品） 

  （开始时间：2026.3.12    23:00  结束时间：2026.3.13	6：00）
演示图片：
<img width="1920" height="911" alt="541164971d47577b66fcc93bfd74d336" src="https://github.com/user-attachments/assets/40e9dab9-f40b-4e6f-9ee1-cc490cfeb75b" />
<img width="1920" height="911" alt="1243c540343d16976afdf33a5694668c" src="https://github.com/user-attachments/assets/35c7bb02-5abd-4deb-851b-a8bf2d2219c7" />


一 、准备工作：安装Python（编程语言环境）、cursor（AI工具、智能代码编辑器）、Git（版本管理工具）、VSCode（备用编辑器）

1 、安装Python（编程语言环境）、cursor（AI工具、智能代码编辑器）、Git（版本管理工具）、VSCode（备用编辑器）
  Python安装（Windows）：https://www.python.org/downloads/windows/
   1 . 安装版本：3.10.11
  2 . 安装过程：安装时勾选“Add Python to PATH”之后一直点击下一步，（个人习惯按在自己知道的目录所以选择更改文件安装位置）

Cursor安装：https://cursor.com/
							1 . 安装版本：当时最新版（我安装的是2.6.18）
							2 . 安装过程：同意协议 —— 一直点击下一步 ——安装
	
Git安装（Windows）：https://git-scm.com/install/windows
							1 . 安装版本：当时最新版（我安装的是2.53.0）
							2 . 安装过程：同样一直点击下一步

VSCode安装：https://code.visualstudio.com/Download
							1 . 安装版本：当时最新版（1.111.0）
							2 . 安装过程：同样一直点击下一步
2 、安装验证
		1 . 打开命令提示符（Win+R，输入cmd）
		2 . 输入python --version和git --version看到版本号说明安装成功
	
3 、注册AI平台获取API密钥（我选择的是智谱AI）
		1 . 进入智谱AI：https://open.bigmodel.cn/
		2 . 注册完成后在个人中心里选择API Key 添加新的API Key ，创建完成后在API Key中复制保存

二 、项目创建和数据准备
	
1 、创建项目文件夹结构
			操作方法：先在桌面创建一个文件夹，重命名为chinese_culture_ai——然后进入文件夹中再逐一创建以下子文件夹——最后在文件夹中创建文本文档，重命名（包括扩展名）（其中app.py、项目说明.md都是创建的文本文档改的）（第一开始我想创建中文文件夹  但是一直运行出错所以改成英文文件夹）

chinese_culture_ai/
src/               	 	  （源代码）
 app.py				        	（核心代码文件）
requirements.txt	  		（类似购物清单，需要哪些第三方库）
.env					         	（类似密码本，如API秘钥、密码、数据库连接等，只能自己知道）
.gitignore			      	（GitHub的屏蔽清单， Git类似搬家公司，写的代码等于要搬走的家具，临时文件=垃圾，密码=私人物品，大文件=大件物品，.gitignore就是搬家清单上不让搬的东西）
data/                   （这个是知识库数据，所有的文本内容或者图片内容或者其他视频之类的在这里）
 knowledge_base.txt			（这个里面是文字内容）
images/                 （全部JPG格式（可以png但是所有图片格式需要一样））


2 、准备核心数据文件
			在data/texts/knowledge_base.txt（我在里面写的是关于神话传说人物和奇珍异兽还有古建筑）
3 、准备图片（后续我加了个AI制图功能，但是有个BUG 在搜索文字时 右侧显示的图片为我本地图片 不会改成AI生成的图片，偷懒把所有本地图片删了全部改为AI制图了...）
			把搜索到的图片放在data/images/文件夹中
	
三 、 用Cursor编写代码

1 、 用Cursor生成主程序，打开Cursor登录，我选择GitHub登录，因为我英文不好，所以先使用Ctrl+Shift+X  进入扩展页面在里面搜Chinese简体中文安装
2 、先点击Open project  选择刚才创建的文件夹中华文化AI助手，然后点击左侧的app.py，再使用快捷键Ctrl+K打开对话框告诉AI提示词

我的提示词是：（ChromaDB我一直报错就换了  ，此行不是提示词不要复制粘贴）

请帮我创建一个中华文化智能问答系统，用 Python 编写。

功能要求：
1. 读取 data/texts/knowledge_base.txt 文件
2. 文件格式是：**标题** 开头，然后有 image: 图片名.jpg，description: 描述
3. 使用智谱AI的API（Zhipu AI），需要从 .env 文件读取 API 密钥
4. 用 ChromaDB 做向量数据库，保存到 vector_db/ 文件夹
5. 用 Gradio 做网页界面，左边显示文字回答，右边显示相关图片
6. 用户提问时，先搜索知识库，找到最相关的信息，然后调用智谱AI生成回答，并显示对应的图片

请写完整的 app.py 代码，要有详细的中文注释。

四 、 创建配置文件
	1 、 把需要的库写在requirements.txt中，我是让cursor帮我写的
	2 、 把从智谱AI复制的密钥粘贴在.env里，（注意从智谱AI复制的密钥是API Key ID和secret连在一起的 ，需要的只有小数点之前的，具体格式为ZHIPUAI_API_KEY=你的智谱AI_API密钥）

五 、 安装依赖

在cursor中打开终端，可以在软件上面点击也可以使用快捷键（Ctrl+`），
	在终端里先输入cd src，再输入pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  （注意中间有空格

六 、 运行程序

在终端中输入python app.py  如果一切正常可以看见一个网址 ，然后打开浏览器 在网址中输入http://localhost:7860，就可以使用了。


以上内容为前期预想，后面在使用cursor中遇到了一些问题
一 、知识库文件路径找不到：因为文件夹少了一个字母 识别不到。已改
二 、HuggingFace连接失败：进不去网站，后面尝试了去镜像网站下载模型BAAI/bge-small-zh-v1.5，但是没找到下载的地方所以改成完全使用智谱AI了
三 、报错无法解析一些库：因为之前下载了一个python14的版本，在把这个版本卸载之后就好了
四 、ChromaDB报错：好像是不兼容，我就换成了FAISS，打开终端输入pip install faiss-cpu ，然后再把之前Chroma的代码都换成FAISS   ，

这些大概就是过程中遇到的一些问题 ，后面提示可以制作临时公网链接，但是我打不开那个网址，而且cursor的免费额度也没了  所以就这样了，
还有一些BUG没改：在知识问答界面中  输入不在库里的问题会回复其他的内容   乱生图

不知道为什么调整的间距都挤在一起了，这个问答做的一般，等我再学学   学成归来之后好好改下
  <img width="2005" height="989" alt="image" src="https://github.com/user-attachments/assets/509993e2-debf-42ca-a711-92635b1b8c23" />








