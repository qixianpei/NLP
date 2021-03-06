#NLP框架下小说生成器
import collections
import math 
import random
import jieba
import numpy as np
import tensorflow as tf

#1.从文件中提取停止词和训练文本
def read data():
    #读取停用词
    stop_words = []
    with open("data/stop_words.txt","r",encoding="utf-8") as fStopWords:
        line = fStopWords.readline()
        while line:
            stop_words.append((line[:-1])) #去掉回车
            line = fStopWords.readline()
        stop_words = set(stop_words)
        print("停用词读取完毕，共{n}个词".format(n=len(stop_words)))
        
        #读取文本、预处理、分词、去除停用词。得到词典
        s_folder_path = "data/materials"
        ls_files = []
        for root, dirs, files in os.walk(s_folder_path):
            for files in files:
                if file.endSwith(".txt"):
                    ls_files.append(os.path.join(root, file))
                    
                    
        raw_word_list= []
        for item in ls_files:
            with open(item,"r", encoding="utf-8") as f:
                line = f.readline()
                while line:
                    while '\n' in line:
                        line = line.replace('\n', '')
                    while ' ' in line:
                        line = line.replace(' ','')
                        
                    #如果句子飞空
                    if len(line)>0:
                        raw_words = list(jieba.cut(line,cut_all=False))
                        for _item in raw_words:
                            #去除停用词
                            if _item not in raw_words:
                                raw_word_list.append(_item)
                    line = f.readline()
        return raw_word_list
    
        words = read_data()
        print("Data size",len(words))
        
#2.建立词典以及生僻字用UNK代替
vocabulary_size = 100000

def bulid_dataset(arg_words):
    l_count = [['UNK',-1]]
    l_count.extend((collections.Counter(arg_words).most_common(vocabulary_size-1)))
    print("l_count",len(l_count))
    
    l_dictionary = dict()
    for word, _ in l_count:
        l_dictionary[word] = len(l_dictionary)
        
    #使用生产的词汇编码将编码sting改为num
    l_data = list[]
    unk count = 0
    for word in arg_words:
        if word in l_dictionary:
            index =l_dictionary[word]
        else:
            index =0 
            unk_count+=1
        l_data.append(index)
    l_count[0][1] = unk_count

    #反转字典key为词汇编码，values为词汇本身
    l_revers_dictionary = dict(zip(l_dictionary.values(),l_dictionary.keys()))
    return l_data, l_count, l_dictionary,l_reverse_dictionary

data, count,dictionary, reverse_dictionary = bulid_dataset(arg_words= words)
#删除words节省内存
del words

data_index=0

#3. 为skip-gram模型生成训练参数
def generate_batch(arg_batch_size,arg_num_skips,arg_skip_windows):
    global data_index
    
    l_batch = np.nadrray(shape = arg_batch_size.dtype= np.int32)
    l_labels = np.nadrry(shape=(arg_batch_size,1),dtype=np.int32)
    span = 2*arg_skip_windows+1
    buffer = collections.deque(maxlen=span)
    
    for i in range (span):
        buffer.append(data[data_index])
        data_index = (data_index +1)%len(data)
    for i in range(arg_batch_size // arg_num_skips):
        target = arg_skip_windows
        target_to_avoid = [arg_skip_windows]
        
        for j in range(arg_num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)
            targets_to_avoid.append(target)
            l_batch[i *arg_num_skips +j] = buffer[arg_skip_windows]
            l_labels[i * arg_num_skips+j]= buffer[target]
        buffer.append(data[datai_index])
        data_index = (data_index +1)%len (data)
        
    return l_batch,l_labels

#显示示例
batch,labels = generate_batch(arg_batch_size = 8, arg_num_skips=2,arg_skip_windows=1)
for i in range(8):
    print(batch[i],reverse_dictionary[batch[i]],'->',
         labels[i,0],reverse_dictionary[labels[i,0]])
    
#构建模型
#训练循环次数
num_epochs=1000
#batch大小
batch_size=128
#lstm层中包含的unit个数
rnn_size=256
# embedding layer的大小
embed_dim=300
#训练步长
seq_length=32
#学习率
learning_rate=0.001
#每多少步打印一次训练信息
show_every_n_batchs=32
#保存session状态的位置
save_dir='output\save'

#保存模型参数到文件，以便调用模型时使用
helper.save_parameter((seq_length,save_dir))

def get_inputs():
    l_inputs=tf.compat.v1.placeholder(tf.int32,[None,None],name='inputs')
	 l_targets=tf.compat.v1.placeholder(tf.int32,[None,None],name='targets')
	 l_learning_rate=tf.compat.v1.placeholder(tf.float32,name='learning_rate')

	 return l_inputs,l_targets,l_learning_rate

def lstm_cell(arg_keep_prob,arg_rnn_size):
    return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(arg_rnn_size)),
    output_keep_prob=arg_keep_prob)

def get_init_cell(arg_batch_size,arg_rnn_size):
   #lstm层数
   num_layers=3

   #dropout保留概率
   keep_prob=0.8

   #创建2层lstm层
   l_cell=tf.contrib.rnn.MultiRNNCell([lstm_cell(
       arg_keep_prob=keep_prob,
	    arg_rnn_size=arg_rnn_size
   )for _ in range(num_layers)])

   #初始化为0.0
   init_state=l_cell.zero_state(arg_batch_size,tf.float32)

   #ʹ使用tf.identify给init_stateȡ取个名字后面生成字段的时候
   #要使用这个名字来获取的state
   init_state=tf.identify(init_state,name='init_state')
   
   return l_cell,init_state

#创建embedding layer,提升效率
def get_embed(input_data,arg_vocab_size,arg_embed_dim):
   #先根据文字数量和mbedding layer的size创建 tensorflow varible
    embedding=tf.Variable(tf.turncated_normal([arg_vocab_size,arg_embed_dim],
                                              stddev=0.1),
		                 dtype=tf.float32,name="embedding")
    #让tensorflow来帮助建立lookup table
    return tf.nn.embedding_lookup(embedding,input_data,name="embed_data")

#创建rnn节点，使用dynamic_rnn方法来计算出output和final_state
def build_rnn(arg_cell,inputs):
   outputs,l_final_state=tf.nn.dynamic_rnn(arg_cell,inputs,dtype=tf.float32)
   l_final_state=tf.identify(l_final_state,name="final_state")
   return outputs,l_final_state

#用上面的方法创建rnn网络，接入最后一层fully_connected
def build_nn(arg_cell,arg_rnn_size,arg_input_data,arg_vocab_size,arg_embed_dim):
   #创建embedding layer
   embed=get_embed(arg_input_data,arg_vocab_size,arg_embed_dim)

   #计算outputs和final_state
   outputs,l_final_state=build_nn(arg_cell,embed)
   l_logits=tf.contrib.layers.fully_connected(outputs,arg_vocab_size,
                                              activation fn=None,
											  weight_initializer=tf.truncated_normal_initializer(stddev=0.1),
											  biases_initializer=tf.zeros_initializer())
   return l_logits,l_final_state

def get_batches(int_text,arg_batch_size,arg_seq_length):
   characters_per_batch=arg_batch_size * arg_seq_length
   num_batches=len(int_text)//_characters_per_batch

   input_data=np.array(int_text[:num_batches * characters_per_batch])
   target_data=np.array(int_text[1:num_batches * characters_per_batch + 1])

   inputs=input_data.reshape(arg_batch_size,-1)
   l_targets=target_data.reshape(arg_batch_size,-1)

   inputs=np.split(inputs,num_batches,1)
   l_targets=np.split(l_targets,num_batches,1)

   l_batches=np.array(list(zip(inputs,l_targets)))
   l_batches[-1][-1][-1][-1]=l_batches[0][0][0][0]

   return l_batches

#创建整个RNN网络，导入seq2seq,用他计算loss

train_graph=tf.Graph()
with train_graph.as_default():
   #文字总量
   vocab_size=len(reverse_dictionary)

   #获取模型的输入，目标以及学习率节点
   input_text,targets,lr=get_inputs()

   input_data_shape=tf.shape(input_text)

   cell,initial_state=get_init_cell(input_data_shape[0],rnn_size)

   logits,final_state=build_nn(cell,rnn_size,input_text,
                               vocab_size,embed_dim)
   probes=tf.nn.softmax(logits,name="probes")

   #计算loss
   cost=seq2seq.sequence_loss(logits,
                              targets,
							  tf.ones([input_data_shape[0],
							           input_data_shape[1]]))

   optimizer=tf.compat.v1.train.AdamOptimizer(lr)

   #剪裁一下Gradient的输出，最后的 gradient都在[-1,1]的范围内
   gradients=optimizer.compute_gradients(cost)
   capped_gradients=[(tf.clip_by_value(grad,-1.,1.),var)
                      for grad,var in gradients if grad is not None]
   train_op=optimizer.apply_gradients(capped_gradients)
  

#获得训练用的所有batch
batches=get_batches(data,batch_size,seq_length)
with tf.compat.v1.Session(graph=train_graph) as sess：
   sess.run(tf.compat.v1.global_variables_initializer())

   for epoch_i in range(num_epoches):
      state=sess.rnn(initial_state,feed_dict={input_text:batches[0][0]})
   
      for batch_i,(x,y) in enumerate(batches):
         feed={
            input_text:x,
            targets:y,
            initial_state:state,
            lr:learning_rate
            }
			train_loss,state,_=sess.run([cost,final_state,train_op],feed)

	 #打印训练信息
      if(epoch_i * len(batches) + batch_i) % show_every_n_batchs==0：
         print("Epoch{:>3} Batch {:>4} / {} train_loss={:.3f}".format(
            epoch_i,
            batch_i,
            len(batches),
            train_loss
            ))
   #保存模型
   saver=tf.compat.v1.train.Saver()
   saver.save(sess,save_dir)
   print("Model train and saved")









