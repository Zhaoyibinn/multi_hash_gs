# GS Compress(Multi-Hash) 
本仓库基于tinycudann的多分辨率哈希，采用多分辨哈希和MLP来减小GS的存储容量

用MLP合成代替球谐函数的45维

## Notice
目前已实现用MLP代替45维的球谐，产生结果是放在result里面的plt和pth，格式参考Compact GS,可以吧我们这里压缩好的放进Compact里面进行render

环境配置也与Compact GS相同（主要就是tinycudann，gs都可以不装）

https://github.com/maincold2/Compact-3DGS