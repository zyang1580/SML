# SML
  This is an implementation for our SIGIR 2020 paper: How to Retrain Recommender System? A SequentialMeta-LearningMethod.
  ## Requirements
  pytorch >= 1.2

  numpy

  ## Parameters

  + --MF_lr: learning rate for $\hat(w)_t$
  + --TR_lr: learning rate for Transfer
  + --l2： $\lambda_1$ in paper
  + --TR_l2: $\lambda_2$ in paper
  + --MF_epochs: epochs of learning MF $\hat(w)_t$  (line 6 in Alg 1 in paper)
  + --TR_epochs: epochs of learning Transfer $\theta$ (line 9 in Alg 1 in paper)
  + --multi_num: stop condition  (line 4 in Alg 1 in paper)
  + others: read help, or "python main_yelp.py --help"

  ## Dataset
  Save as array,
  + train: period_num.npy (user_id,item_id)
  + test:  period_num.npy (user_id,pos_item_id, neg_item_id)
  + Dataset URL: https://rec.ustc.edu.cn/share/1e40a9b0-c80a-11ea-b178-77f793cf5b55 (This is Yelp, Adressa will be uploaded later)
  
  ## Examples

  ### Yelp

  `nohup python main_yelp.py --MF_epochs=1 --TR_epochs=1 --multi_num=10 > yelp_log.out &`

  or

  `python main_yelp.py --MF_epochs=1 --TR_epochs=1 --multi_num=10`

  ### Adressa
  `nohup python main_news.py --MF_epochs=2 --TR_epochs=2 --multi_num=7 > yelp_log.out &`
  
  for Adressa, if we set the ***MF_epochs and TR_epochs*** same to paper (=1) , we can also get a similar performance if we adjust ***multi_num***.



