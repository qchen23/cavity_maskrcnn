
ckp=$1
t=$2

cat `find $ckp -name "cm-$2.txt"` > $ckp/all_cm.txt
cat $ckp/all_cm.txt | \
awk '{print $(NF-6), $(NF-3), $NF}' | \
sed 's/,//g' | \
awk '{p+=$1;r+=$2;f1+=$3}END{printf "precision = %lf, recall = %lf, f1_score = %lf\n", p/NR, r/NR, f1/NR}'

# echo "bbx"
cat `find $ckp -name "bbx-cm-$2.txt"` > $ckp/all_bbx_cm.txt
cat $ckp/all_bbx_cm.txt | \
awk '{print $(NF-6), $(NF-3), $NF}' | \
sed 's/,//g' | \
awk '{p+=$1;r+=$2;f1+=$3}END{printf "precision = %lf, recall = %lf, f1_score = %lf\n", p/NR, r/NR, f1/NR}'


# cat $ckp/all_cm.txt | \
# awk '{print $3, $6, $9, $12}' | \
# sed 's/,//g' | \
# awk '{tp+=$1;tn+=$2;fp+=$3;fn+=$4}END{p=tp/(tp+fp);r=tp/(tp+fn);printf "precision = %lf, recall = %lf, f1_score = %lf\n", p, r, 2*p*r/(p+r)}'
