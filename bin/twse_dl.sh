cd /tmp2/pcchien/twse

wget -qOa "https://www.twse.com.tw/exchangeReport/MI_INDEX?type=ALLBUT0999"
date=`jq -r .date a`
if test -f raw/$date.json
then
	rm a
	exit
fi
traded_stock=`jq -r '.data9[][0]' a`
jq -r '.date as $d|.data9[]|map(gsub(",";""))|[$d]+.|@csv' a | tr -d \" |
	awk -F, -v OFS=, '{print >> "stock/" $2 ".csv"}'
mv a raw/$date.json

wget -qOa "https://www.twse.com.tw/indicesReport/MI_5MINS_HIST"
mv a index/`jq -r .date[:6] a`.json

mkdir -p macd_cross/.tmp
parallel bash -c '"python3 bin/ema.py stock/{}.csv 5 8 && touch macd_cross/.tmp/{.}"' ::: $traded_stock
echo '# DIF5-8 > 0' > macd_cross/$date.txt
grep -wf <(ls macd_cross/.tmp) meta/id_to_name.txt >> macd_cross/$date.txt
rm macd_cross/.tmp/*
echo '# DIF12-26 > 0' >> macd_cross/$date.txt
parallel bash -c '"python3 bin/ema.py stock/{}.csv 12 26 && touch macd_cross/.tmp/{.}"' ::: $traded_stock
grep -wf <(ls macd_cross/.tmp) meta/id_to_name.txt >> macd_cross/$date.txt
rm -r macd_cross/.tmp

awk -v OFS='\t' -F'\t' 'ARGIND == 1 {a[$1]=$2; next} NF == 1 {print; next} {$3=$2 in a?a[$2]:0;print}' <(curl -s "https://raw.githubusercontent.com/dibery/positioncalculation/master/record/$date.txt") macd_cross/$date.txt | sponge macd_cross/$date.txt

git add index macd_cross raw
git commit -m 'Info update' 2> /dev/null
git push origin master 2> /dev/null
mail -s "$date MACD cross" dibery@ntu.im < macd_cross/$date.txt
