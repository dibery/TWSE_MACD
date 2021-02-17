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

mail -s "$date MACD cross" dibery@ntu.im < macd_cross/$date.txt
