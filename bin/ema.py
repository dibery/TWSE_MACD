# Usage: python3 sys.argv[0] price_file.csv fast_ema slow_ema
import pandas, sys
s = pandas.read_csv(sys.argv[1], header=None)
try:
    s = (s[7] + s[8] + s[9] * 2) / 4
except:
    exit(1)
dif = s.ewm(span=int(sys.argv[2])).mean() - s.ewm(span=int(sys.argv[3])).mean() >= 0
exit(0 if not dif.iloc[-4:-1].any() and dif.iloc[-1] > 0 else 1)
