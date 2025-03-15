# $1: number of shots
# $2: dataset (flower/food101)

# print $1
echo "Number of shots: $1"
echo "Dataset: $2"

python main.py --cfg cfg/asso_opt/$2/$2_$1shot_fac.py --work-dir exp/asso_opt/$2/$2_$1shot_fac --func asso_opt_main