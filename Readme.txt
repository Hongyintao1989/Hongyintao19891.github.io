1、普通模块消融：
训练指令：
python run_all_configs.py
测试指令：
run_all_tests.py



测试指令：

python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_base.pth"

python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm.pth" --use_saspp --use_damm
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm_dropblock.pth" --use_damm --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm_gcnet.pth" --use_damm --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_dropblock_gcnet.pth" --use_dropblock --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_dropblock.pth" --use_saspp --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_gcnet.pth" --use_saspp --use_gcnet

python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm.pth" --use_damm
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp.pth" --use_saspp
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_gcnet.pth" --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_dropblock.pth" --use_dropblock


python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm_dropblock_gcnet.pth" --use_damm --use_dropblock --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm_dropblock.pth" --use_saspp --use_damm --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm_gcnet.pth" --use_saspp --use_damm --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_dropblock_gcnet.pth" --use_saspp --use_dropblock --use_gcnet

python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm_dropblock_gcnet.pth" --use_saspp --use_damm --use_dropblock --use_gcnet

checkppoint0

更新：
1、普通模块消融：
训练指令：
python run_all_configs.py
测试指令：
run_all_tests.py

或者如下：
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm.pth" --use_saspp --use_damm
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm_dropblock.pth" --use_damm --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm_gcnet.pth" --use_damm --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_dropblock_gcnet.pth" --use_dropblock --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_dropblock.pth" --use_saspp --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_gcnet.pth" --use_saspp --use_gcnet

python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm.pth" --use_damm
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp.pth" --use_saspp
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_gcnet.pth" --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_dropblock.pth" --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_base.pth"

python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm_dropblock_gcnet.pth" --use_damm --use_dropblock --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm_dropblock.pth" --use_saspp --use_damm --use_dropblock
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm_gcnet.pth" --use_saspp --use_damm --use_gcnet
python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_dropblock_gcnet.pth" --use_saspp --use_dropblock --use_gcnet


python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm_dropblock_gcnet.pth" --use_saspp --use_damm --use_dropblock --use_gcnet

2、分支消融：
训练：run_all_configs-fenzhixiaorong.py
测试：run_all_tests-fenzhixiaorong.py

3、Q-learning搜索最优损失函数：
强化学习搜索文件：rl_loss_search.py
训练：python train_chase_loss_search.py --use_saspp --use_damm --use_dropblock --use_gcnet --search_loss --method rl
      python train_chase_loss_search.py --use_saspp --use_damm --search_loss --method rl --generations 15 --validation_epochs 25
测试：python eval_chase.py --use_saspp --use_damm --use_dropblock --use_gcnet --loss_config "D:\Test-1\Chase-DB1\Unet-demo copy428\Unet-demo -copy0507新分支改进\search_results\saspp_damm_dropblock_gcnet_20250511_001948\search\best_loss_config.json" --weight_path "D:\Test-1\Chase-DB1\Unet-demo copy428\Unet-demo -copy0507新分支改进\search_results\saspp_damm_dropblock_gcnet_20250511_001948\checkpoint\best_model.pth"


遗传算法：loss_search.py 
  训练：python train_chase_loss_search.py --use_saspp --use_damm --use_dropblock --use_gcnet --search_loss
  测试：python eval_chase.py --use_saspp --use_damm --use_dropblock --use_gcnet --loss_config "D:\Test-1\Chase-DB1\Unet-demo copy428\Unet-demo -copy428\search_results\saspp_damm_dropblock_gcnet_20250506_204440\search\best_loss_config.json" --weight_path "D:\Test-1\Chase-DB1\Unet-demo copy428\Unet-demo -copy428\search_results\saspp_damm_dropblock_gcnet_20250506_204440\checkpoint\best_model.pth"
  