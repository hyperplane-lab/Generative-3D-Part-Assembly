# Generative 3D Part Assembly via Dynamic Graph Learning

This is the implementation of NeurIPS 2020 paper "Generative 3D Part Assembly via Dynamic Graph Learning" created by 
<a href="https://jialeihuang.github.io/" target="_blank">Jialei Huang*</a>, <a href="https://championchess.github.io/" target="_blank">Guanqi Zhan*</a>, <a href="https://fqnchina.github.io/" target="_blank">Qingnan Fan</a>, <a href="https://cs.stanford.edu/~kaichun/" target="_blank">Kaichun Mo</a>, <a href="https://linsats.github.io/" target="_blank">Lin Shao</a>, <a href="https://cfcs.pku.edu.cn/baoquan/" target="_blank">Baoquan Chen</a>, <a href="https://geometry.stanford.edu/member/guibas/index.html" target="_blank">Leonidas Guibas</a> and <a href="https://zsdonghao.github.io/" target="_blank">Hao Dong</a>.

![image1](./images/image1.png)

The proposed dynamic graph learning framework. The iterative graph neural network backbone takes a set of part point clouds as inputs and conducts 5 iterations of graph message-passing for coarse-to-fine part assembly refinements. The graph dynamics is encoded into two folds, (a) reasoning the part relation (graph structure) from the part pose estimation, which in turn also evolves from the updated part relations, and (b) alternatively updating the node set by aggregating all the geometrically-equivalent parts (the red and purple nodes), e.g. two chair arms, into a single node (the yellow node) to perform graph learning on a sparse node set for even time steps, and unpooling these nodes to the dense node set for odd time steps. Note the semi-transparent nodes and edges are not included in graph learning of certain time steps.

- [paper link](https://arxiv.org/pdf/2006.07793.pdf)
- [project page](https://hyperplane-lab.github.io/Generative-3D-Part-Assembly/)


## File Structure

This repository provides data and code as follows.


```
    data/                       # contains PartNet data
        partnet_dataset/		# you need this dataset only if you  want to remake the prepared data
    prepare_data/				# contains prepared data you need in our exps 
    							# and codes to generate data
    	Chair.test.npy			# test data list for Chair (please download the .npy files using the link below)
    	Chair.val.npy			# val data list for Chair
    	Chair.train.npy 		# train data list for Chair
    	...
        prepare_shape.py/				    # prepared data
    	prepare_contact_points.py			# prepared data for contact points
    	
    exps/
    	utils/					# something useful
    	dynamic_graph_learning/	# our experiments code
    		logs/				# contains checkpoints and tensorboard file
    		models/				# contains model file in our experiments
    		scripts/			# scrpits to train or test
    		data_dynamic.py		# code to load data
    		test_dynamic.py  	# code to test
    		train_dynamic.py  	# code to train
    		utils.py
    environment.yaml			# environments file for conda
    		

```

This code has been tested on Ubuntu 16.04 with Cuda 10.0.130, GCC 7.5.0, Python 3.7.6 and PyTorch 1.1.0. 

Download the [pre-processed data](http://download.cs.stanford.edu/orion/genpartass/prepare_data.zip) for the .npy data files in file prepare_data/


## Dependencies

Please run
    

        conda env create -f environment.yaml
        . activate PartAssembly
        cd exps/utils/cd
        python setup.py build

to install the dependencies.

## Quick Start

Download [pretrained models](http://download.cs.stanford.edu/orion/genpartass/checkpoints.zip) and unzip under the root directory.

### Train the model

Simply run

        cd exps/dynamic_graph_learning/scripts/
        ./train_dynamic.sh
        
### Test the model

modify the path of the model in the test_dynamic.sh file

run

        cd exps/dynamic_graph_learning/scripts/
        ./test_dynamic.sh

## Questions

Please post issues for questions and more helps on this Github repo page. We encourage using Github issues instead of sending us emails since your questions may benefit others.

## Maintainers
@Championchess 
@JialeiHuang


## Citation

    @InProceedings{HuangZhan2020PartAssembly,
        author = {Huang, Jialei and Zhan, Guanqi and Fan, Qingnan and Mo, Kaichun and Shao, Lin and Chen, Baoquan and Guibas, Leonidas and Dong, Hao},
        title = {Generative 3D Part Assembly via Dynamic Graph Learning},
        booktitle = {The IEEE Conference on Neural Information Processing Systems (NeurIPS)},
        year = {2020}
    }

## License

MIT License

## Todos

Please request in Github Issue for more code to release.

