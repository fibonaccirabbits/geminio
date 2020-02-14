source /opt/miniconda3/bin/activate tensorflow2
echo sourced conda-env tensorflow2
echo 'running: nohup python size_trainer.py > size_trainer.out &'
nohup python size_trainer.py > size_trainer.out &


