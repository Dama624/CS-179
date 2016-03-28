Hong Joon Park
CS179 Problem Set 5

* What is the latency of classifying a single batch? How does this change with
  batch size?

For batch size = 32:
Latency of classifying a batch: 0.053760 ms

For batch size = 2048:
Latency of classifying a batch: 0.054752 ms

For batch size = 1024:
Latency of classifying a batch: 0.055104 ms

For batch size = 4096:
Latency of classifying a batch: 0.059392 ms

The latency of classifying a batch does not seem to change drastically with batch size.
It (logically) seems to go up very minimally as batch size increases.


* What is the throughput of cluster.cc in reviews / s? How does this change with
  batch size?

For batch size = 32:
Latency of processing 1 batch: 3.775712 ms
32 reviews per 3.775712 ms = 8 reviews per ms
= 8,000 reviews per second

For batch size = 2048:
Latency of processing 1 batch: 59.909344 ms
1 batch has size 2048 (reviews)
2048 reviews per 59.909344 ms = 34 reviews per ms
= 34,000 reviews per second

For batch size = 1024:
Latency of processing 1 batch: 34.292866 ms
1024 reviews per 34.292866 ms = 31 reviews per ms
= 31,000 reviews per second

For batch size = 4096:
Latency of processing 1 batch: 126.516701 ms
4096 reviews per 126.516701 ms = 32 reviews per ms
= 32,000 reviews per second

The throughput seems to level off at some point, but there is a very notable difference in
throughput for very small batch sizes.
For small batch sizes, the throughput is much lower compared to larger batch sizes.


* Do you think you could improve performance using multiple GPUs? If so, how?
  If not, why not?
I don’t think it would improve performance that much; in this code, the clusters are being
constantly updated. Therefore, data would be being read and written among the multiple GPUs
constantly. There is a penalty to communicating between GPUs, thus there may not be as
great a performance increase from using multiple GPUs.