#用生成的模型生成小说内容
#生成的文本长度
gen_length = 500

#文章开头的字，指定一个就行，但是该字必须存在于训练的词汇表里
prime_word = "和"

loaded_graph = tf.Graph()
with tf.compat.vl.Session(graph=loaded_graph) as sess:
	#加载保存过的Session
	loader = tf.compat.vl.train.import_meta_graph(load_dir+".meta")
	loader.restore(sess,load_dir)

	#通过名称获取缓存的tensor
	input_text,intial_state,final_state.probes =get_tensors(loaded_graph)

	#准备开始生成文本
	gen_sentences= [prime_word]
	prev_state = sess.run(intial_state,feed_dict={input_text:np.array([[1]])})

	#开始生成文本
	for n in range(gen_length):
		dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
		dyn_seq_length = len(dyn_input[0])

		probabilities,prev_state = sess.run(
			[probes,final_state],
			feed_dict={input_text:dyn_input,intial_state:prev_state})

		probes_array = probabilities[0][dyn_seq_length-1]
		pred_word = pick_word(probes_array,int_to_vocab)
		gen_sentences.append(pred_word)

	#将标点符号还原
	novel = ''.join(gen_sentences)
	for key,token in token_dict.items():
		ending = ''if key in['\n','(','"'] else ''
		novel = novel.replace(token,key)

	print(novel)