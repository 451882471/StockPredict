DDPG版本（wudigushi）：
	在158行main函数有个train的参数，train=True则训练，Train=False则评价。
	在136行和137行中分别设置训练好的模型和数据的路径
	在终端输入python wudigushi/wuditrain.py并运行
	训练好的模型放在wudigushi/ckpt中


TD3版本（TD3gupiao）：
	直接在终端运行python TD3gupiao/TD3_Train.py
	trainlog放在train_log文件夹中


注：由于时间的关系，该版本还存在着许多不完善的地方，以后会逐步地进行版本迭代。
