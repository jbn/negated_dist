Negated Scipy Distributions
===

```python
from scipy.stats import norm
from negated_dist import NegatedDist

X = NegatedDist(norm(1))
print(X.mean())     # => -1.0
print(X.ppf(0.5))   # => -1.0
print(X.std())      # =>  1.0
print(X.moment(1))  # => -1.0
print(X.moment(2))  # =>  2.0
print(X.rvs())      # => ~N(-1, 1)
```
