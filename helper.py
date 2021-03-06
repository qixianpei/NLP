import pickle
#定义符号表
def token_lookup():
	symbols = {"。","，","“","”","；","！","？","（","）","——","\n"}
	tokens ={"P","C","Q","T","S","E","M","I","O","D","R"}
	return dict(zip(symbols,tokens))


#保存预处理数据到制定的二进制文件
def save_data(token,vocab_to_int,int_to_vocab):
	pickle.dump((token,vocab_to_int,int_to_vocab),open('data\prerocess.p','wb'))

#从保存的数据文件加载到内存

def load_data():
	return pickle.load(open('data\prerocess.p',mode='rb'))

#保存模型参数到二进制文件
def save_parameter(params):
	pickle.dump(params,open('data\prerocess.p',wb))


#加载模型参数到内存
def load_parameter():
	return pickle.load(open('data\params.p',mode='rb'))