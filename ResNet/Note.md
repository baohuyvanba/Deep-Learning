- [Tại sao lại là ResNet?](#i-why-resnet)
  - [Vấn đề suy giảm hiệu suất trong mạng học sâu](#1-vấn-đề-suy-giảm-hiệu-suất-trong-mạng-học-sâu)
  - [Vanishing Gradient và tính ổn định số học](#2-vấn-đề-vanishing-gradient-và-tính-ổn-định-số-học)
  - [Hạn chế của mạng Nơ-ron sâu và Giải pháp](#3-hạn-chế-của-mạng-nơ-ron-sâu-và-giải-pháp)
- [Nguyên lý ResNet](#ii-nguyên-lí-resnet)
  - [Lớp hàm số. Hàm số lồng nhau (Nested function classes)](#1-lớp-hàm-số-hàm-số-lồng-nhau-nested-function-classes)
  - [Khối phần dư (Residual Block): Học hàm phần dư thay vì hàm mục tiêu](#2-khối-phần-dư-residual-block-học-hàm-phần-dư-thay-vì-hàm-mục-tiêu)
  - [Kết nối Tắt (Shortcut Connection) và Ánh xạ đồng nhất (Identity Mapping): Đường tín hiệu và Gradient hiệu quả](#3-kết-nối-tắt-shortcut-connection-và-ánh-xạ-đồng-nhất-identity-mapping-đường-tín-hiệu-và-gradient-hiệu-quả)
- [ResNet: Toán học](#iii-resnet-toán-học)
  - [Khối phần dư (Residual Block)](#1-khối-phần-dư-residual-block)
  - [Không gian hàm và tối ưu hóa](#2-không-gian-hàm-và-tối-ưu-hóa)
  - [Ảnh hưởng của Kết nối Tắt đến Gradient và Huấn luyện](#3-ảnh-hưởng-của-kết-nối-tắt-đến-gradient-và-huấn-luyện)
- [Kiến trúc và hoạt động](#iv-kiến-trúc-và-hoạt-động)
  - [Kiến trúc tổng quan](#1-kiến-trúc-tổng-quan)
  - [Cấu trúc khối phần dư: Cơ bản và Bottleneck](#2-cấu-trúc-khối-phần-dư-cơ-bản-và-bottleneck)
  - [Các biến thể ResNet](#3-các-biến-thể-resnet)
  - [Truyền xuôi và Truyền ngược](#4-truyền-xuôi-và-truyền-ngược)
- [Cài đặt và Huấn luyện](#v-cài-đặt-và-huấn-luyện)
  - [Xây dựng Khối phần dư và Module ResNet](#1-xây-dựng-khối-phần-dư-và-module-resnet)
    - [Lớp phần dư - Residual Block](#11-lớp-phần-dư-residual-block)
      - [Giá trị `bias` trong hàm nn.Conv2d](#111-giá-trị-bias-trong-hàm-nnconv2d)
      - [Batch Normalization](#112-batch-normalization)
    - [ResNet Module](#12-resnet-module)
      - [Basic Block](#121-basic-block)
      - [Bottleneck Block](#122-bottleneck-block)
  - [Kiến trúc ResNet hoàn chỉnh](#2-kiến-trúc-resnet-hoàn-chỉnh)
- [Tham khảo](#tham-khảo)



# I. Why Resnet?
- Nhằm giải quyết những thách thức trong việc huấn luyện các mạng nơ-ron sâu: Vanishing Gradient.
- Vấn đề này trở nên ngày càng quan trọng khi các nhà nghiên cứu và kỹ sư muốn xây dựng các mô hình phức tạp hơn để cải thiện hiệu suất trong các tác vụ thị giác máy tính và nhiều lĩnh vực khác.
## 1. Vấn đề suy giảm hiệu suất trong mạng học sâu
- Khi mạng nơ-ron có nhiều lớp hơn (sâu hơn), lý thuyết cho rằng nó sẽ có *khả năng biểu diễn tốt hơn*, cho phép mô hình nắm bắt các đặc trưng phức tạp và cải thiện độ chính xác.
- Tuy nhiên, các mạng học sâu (plain network) gặp phải **vấn đề suy giảm hiệu suất** (Degradation): nghĩa là một mạng sâu hơn lại gặp vấn đề với độ chính xác trên cả tập huấn luyện và kiểm tra.

- Khác với Overfitting, vấn đề này là lỗi huấn luyện xảy ra khi mạng trở nên sâu hơn.
- Nguyên nhân là do vấn đề tối ưu hóa khó khăn hơn khi độ sâu của mô hình tăng lên, đặc biệt với các phương pháp truyền thống như *gradient descent*.
- Một trong những giả thuyết được đưa ra là các mạng nơ-ron sâu gặp khó khăn trong việc học các ánh xạ đồng nhất (identity mappings).
## 2. Vấn đề Vanishing Gradient và tính ổn định số học
- *Nguyên nhân gốc rễ của vấn đề suy giảm hiệu suất* là vấn đề Vanishing/Exploding Gradient.
- Trong quá trình lan truyền ngược, Gradient được tính và lan từ $\text{output}\to\text{input}$. Khi đó, mạng càng sâu, *gradient qua nhiều lớp và nó trở nên quá nhỏ/lớn* - ảnh hưởng quá trình học của mạng.
	- Vanishing (tiêu biến): gradient trở nên quá nhỏ, các lớp ở gần đầu vào nhận được rất ít cập nhập -> các lớp không học/học rất chậm -> giảm hiệu quả việc tăng độ sâu mạng.
	- Exploding (quá lớn): việc cập nhập trọng số quá mạnh, gây mất ổn định huấn luyện.
- Các kỹ thuật khởi tạo trọng số và chuẩn hóa trung gian đã giúp giảm thiểu vấn đề vanishing/exploding gradient đến một mức độ nhất định, nhưng chúng vẫn không thể giải quyết triệt để vấn đề.
## 3. Hạn chế của mạng Nơ-ron sâu và Giải pháp
- Vấn đề: như đã nêu ở trên, giảm hiệu suất với nguyên nhân do hiện tượng vanishing/explodin gradient.
- Giải pháp: xây dựng một kiến trúc mạng nhằm giải quyết vấn đề. ResNet ra đời như một giải pháp đột phá, cung cấp một phương pháp học mới dựa trên **hàm phần dư** và **kết nối tắt**.
# II. Nguyên lí ResNet
- ResNet hoạt động dựa trên một số nguyên lí cốt lõi bao gồm ý tưởng về lớp hàm số lồng nhau, khối phần dư, và kết nối tắt với ánh xạ đồng nhất.
- Các nguyên lý này phối hợp với nhau để tạo ra một kiến trúc mạng dễ huấn luyện hơn, ổn định hơn và hiệu quả hơn khi độ sâu tăng lên.
## 1. Lớp hàm số. Hàm số lồng nhau (Nested function classes)
- Mỗi kiến trúc mạng nơ-ron có thể biểu diễn một tập hợp các hàm số nhất định, được gọi là **lớp hàm số của kiến trúc** đó.
- Khi ta xây dựng một mạng sâu, ta kì vọng các lớp hàm số này sẽ **lồng** với nhau (nested) với các lớp của mạng nông hơn. Tức làm lớp hàm số của mạng sâu hơn sẽ chứa lớp hàm số nông hơn như hàm con.
- Việc này ít nhất về mặt lý thuyết cho phép mạng nơ-ron sâu có khả năng biểu diễn tốt "bằng" các mạng nông và tiềm năng tốt "hơn" khi huấn luyện hợp lý.
- Tuy nhiên, với các mạng nơ-ron thông thường, việc thêm lớp không đảm bảo tính chất lồng nhau này, và việc tối ưu hóa trong không gian hàm số lớn hơn có thể trở nên khó khăn hơn, dẫn đến vấn đề suy giảm hiệu suất.
## 2. Khối phần dư (Residual Block): Học hàm phần dư thay vì hàm mục tiêu
- ResNet giải quyết vấn đề lớp hàm số không lồng nhau và khó tối ưu hóa bằng cách *thay đổi cách mạng học hàm số*.
- Thay vì học hàm mục tiêu $\mathcal{H}(x)$ phức tạp, ResNet khuyến khích học **hàm phần dư (residual)**:
  $$F(x) = \mathcal{H}(x) - x$$
  Tương đương với
  $$\mathcal{H}(x) = F(x)+x$$
  công thức này được thực hiện hóa trong khối phần dư (*cộng với input*).
- **Ưu điểm**:
	- Dễ học ánh xạ đồng nhất:
		- Nếu ánh xạ đồng nhất $\mathcal{H}(x) = x$ là tối ưu thì hàm phần dư tương ứng $F(x) = 0$.
		- Việc học hàm phần dư khi đó sẽ đơn giản hơn học trực tiếp ánh xạ đồng nhất đó bằng một chồng các lớp phi tuyến.
		- Khi đó, khối phần dư chỉ cần đưa trọng số về $0$ - sẽ hoạt động như một ánh xạ đồng nhất.
	- Ổn định huấn luyện:
		- Giảm thiểu Vanishing/Exploding Gradient.
		- Kết nối tắt còn cho phép gradient truyền trực tiếp qua khối mà không bị suy giảm nhiều.
## 3. Kết nối Tắt (Shortcut Connection) và Ánh xạ đồng nhất (Identity Mapping): Đường tín hiệu và Gradient hiệu quả
- Kết nối Tắt là *thành phần then chốt của khối phần dư*, cho phép tín hiệu đầu vào $x$ được bỏ một/hay nhiều lớp và cộng với đầu ra.
- Trong ResNet gốc, kết nối tắt thường thực hiện **ánh xạ đồng nhất (identity mapping)**, tức là đầu ra của kết nối tắt chính là đầu vào $x$ mà không có bất kỳ phép biến đổi nào.
- Công thức toán học của khối phần dư với ánh xạ đồng nhất là
  $$y=F(x)+x$$
  trong đó $F(x)$ là hàm phần dư được học bởi các lớp tích chập.

- Kết nối tắt và ánh xạ đồng nhất đóng vai trò quan trọng trong việc tạo ra một *"đường tắt" cho cả tín hiệu truyền xuôi và gradient truyền ngược*:
	- **Đường dẫn Tín hiệu**:
		- Giúp tín hiệu đầu vào có thể truyền trực tiếp qua nhiều lớp mà không bị biến đổi quá nhiều.
		- Đặc biệt quan trọng trong các mạng rất sâu, nơi mà tín hiệu có thể bị suy giảm hoặc biến mất.
	- **Đường dẫn Gradient**:
		- Tạo ra một đường dẫn trực tiếp cho gradient truyền ngược, giúp giảm thiểu vanishing gradient.
		- Khi gradient truyền qua kết nối tắt, nó không bị nhân với các trọng số của các lớp trung gian, do đó duy trì được độ lớn và hiệu quả hơn trong việc cập nhật trọng số ở các lớp trước đó.
# III. ResNet: Toán học
## 1. Khối phần dư (Residual Block)
- Như đã đề cập ở trên, đầu ra khối phần dư có biểu diễn toán học: 
  $$y = F(x) + x$$
  trong đó **hàm phần dư** $F(x)$ - chuỗi các phép biến đổi phi tuyến, ví dụ: $F(x) = \sigma_2(W_2)\sigma_1(W_1x)$; với $\sigma_i$ là các hàm kích hoạt, $W_i$ là các ma trận trọng số của các lớp. 
- Chú ý rằng việc thêm các kết nối tắt *không* làm tăng số lượng trọng số và *không* làm tăng độ phức tạp, phép cộng phần tử vốn không đáng kể.
## 2. Không gian hàm và tối ưu hóa
- Về mặt không gian hàm số, ResNet giúp tạo ra các lớp hàm số lồng nhau một cách dễ dàng hơn.
- Khi thêm một khối phần dư vào mạng, mạng mới *ít nhất cũng có khả năng biểu diễn các hàm số mà mạng cũ có thể* biểu diễn (bằng cách đặt $F(x)=0$).
- Hơn nữa, mạng mới *có tiềm năng biểu diễn các hàm số phức tạp hơn* nhờ vào các lớp tích chập trong khối phần dư.
- Tính chất lồng nhau này đảm bảo rằng việc tăng độ sâu mạng không làm giảm khả năng biểu diễn và tối ưu hóa của mô hình.
## 3. Ảnh Hưởng của Kết Nối Tắt đến Gradient và Huấn Luyện
- Kết nối tắt có *ảnh hưởng tích cực đến gradient* trong quá trình truyền ngược.
- Xét đạo hàm của đầu ra $y$ theo đầu vào $x$:
  $$\dfrac{\partial y}{\partial x} = \dfrac{\partial (F(x)+x)}{\partial x} = \dfrac{\partial F(x)}{\partial x} + \dfrac{\partial x}{\partial x} = \dfrac{\partial F(x)}{\partial x} + I$$
  trong đó $I$ là ma trận đơn vị.
- Công thức này cho thấy rằng gradient truyền ngược qua khối phần dư có hai thành phần:
	1. Gradient truyền qua nhánh chính (các lớp tích chập): $\dfrac{\partial F(x)}{\partial x}$
	2. **Identity gradient**: $I$ truyền trực tiếp qua kết nối tắt.
		- Identity gradient này đảm bảo rằng gradient không bị suy giảm quá nhiều khi truyền qua nhiều khối phần dư.
		- Hạn chế Vanishing Gradient.
# IV. Kiến trúc và hoạt động
## 1. Kiến trúc tổng quan
- Kiến trúc mạng ResNet thường đi theo một cấu trúc chung, từ một số lớp tích chập và gộp ban đầu, sau đó là chuỗi các module ResNet và kết thúc bằng lớp gộp trung bình toàn cục và kết nối đầu đủ (FC).
	- **Lớp đầu tiên**:
		- Thường là lớp tích chập lớn ($7\times 7$) với stride lớn ($2$) để giảm kích thước đặc trưng ban đầu;
		- Lớp chuẩn hóa Batch;
		- Hàm kích hoạt ReLU;
		- Lớp gộp Max để giảm độ phân giải không gian.
	- **ResNet Modules**:
		- Chia thành nhiều Module, mỗi module gồm 1 số nhất định các khối phần dư.
		- Trong mỗi module, *độ phân giải giảm đi một nửa* (sử dụng stride 2 trong khối phần dư đầu tiên) trong khi *số lượng kênh đặc trưng tăng gấp đôi*.
	- **Lớp cuối cùng**:
		- Lớp gộp trung bình toàn cục để giảm kích thước đặc trưng xuống;
		- Lớp FC để thực hiện tác vụ mong muốn.
## 2. Cấu trúc khối phần dư: Cơ bản và Bottleneck
- ResNet sử dụng hai loại khối phần dư chính:
	1. **Khối phần dư cơ bản**:
		- Gồm 2 lớp tích chập $3\times 3$, mỗi lớp đi kèm 1 lớp chuẩn hóa batch và hàm ReLU.
		- Kết nối tắt thực hiện ánh xạ đồng nhất hoặc projection shortcut nếu cần thay đổi kích thước.
		- Khối cơ bản thường được sử dụng trong các mạng ResNet nông như ResNet-18 và ResNet-34.
	2. **Khối phần dư Bottleneck**:
		- Được thiết kế để giảm độ phức tạp tính toán trong các mạng ResNet sâu hơn.
		- Khối bottleneck bao gồm ba lớp tích chập:
		    - Lớp tích chập $1\times 1$ đầu tiên: Giảm số kênh đầu vào xuống còn bottleneck channels (thường là 1/4 số kênh đầu vào).
		    - Lớp tích chập $3\times 3$: Thực hiện tích chập chính với số kênh đã giảm.
		    - Lớp tích chập $1\times 1$ thứ hai: Tăng số kênh trở lại bằng số kênh đầu vào.
	    - Các lớp tích chập $1\times 1$ hoạt động như "bottleneck" (nút cổ chai), giúp giảm số lượng tham số và phép tính toán trong lớp tích chập $3\times 3$, vốn là lớp tốn kém nhất về mặt tính toán.
	    - Khối bottleneck cũng đi kèm với lớp chuẩn hóa batch và hàm kích hoạt ReLU sau mỗi lớp tích chập, và kết nối tắt tương tự như khối cơ bản.
## 3. Các biến thể ResNet
- ResNet có nhiều biến thể khác nhau, chủ yếu khác nhau về độ sâu (số lớp) và cấu trúc khối phần dư.
- Phổ biến bao gồm ResNet-18, ResNet-34, ResNet-50, ResNet-101, và ResNet-152.
- Số sau dấu gạch ngang thể hiện số lớp có trọng số trong mạng.
- Các mạng ResNet sâu hơn thường sử dụng khối bottleneck để giảm độ phức tạp tính toán, trong khi các mạng nông hơn có thể sử dụng khối cơ bản.
## 4. Truyền xuôi và Truyền ngược
- **Truyền Xuôi (Forward Pass)**:
	- Đầu vào được truyền qua lớp tích chập đầu tiên, sau đó qua các mô-đun ResNet liên tiếp.
	- Trong mỗi khối phần dư, đầu vào được biến đổi bởi các lớp tích chập để tạo ra hàm phần dư, sau đó hàm phần dư được cộng với đầu vào ban đầu thông qua kết nối tắt.
	- Kết quả được đưa qua hàm kích hoạt ReLU và trở thành đầu ra của khối.
- **Truyền Ngược (Backward Pass)**:
	- Hàm mất mát được tính toán dựa trên đầu ra của mạng và nhãn thực tế.
	- Gradient của hàm mất mát được tính toán và truyền ngược từ lớp đầu ra về lớp đầu vào, sử dụng quy tắc dây chuyền (chain rule).
	- Kết nối tắt đóng vai trò quan trọng trong việc truyền gradient hiệu quả, giúp giảm thiểu vanishing gradient và cho phép huấn luyện các mạng sâu một cách ổn định.
	- Các thuật toán tối ưu hóa (ví dụ, SGD, Adam) được sử dụng để cập nhật trọng số của mạng dựa trên gradient tính toán được.
# V. Cài đặt và Huấn luyện
## 1.  Xây dựng Khối phần dư và Module ResNet
- Để cài đặt ResNet, cần xây dựng các thành phần cơ bản là khối phần dư và mô-đun ResNet.
- Dưới đây là các bước cài đặt chi tiết:
### 1.1 Lớp phần dư - Residual Block
```python
import torch
import torch.nn as nn

class ResidualNetwork(nn.Module):
	def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
		super(ResidualBlock, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels, out_channels, kerner_size=3, stride=stride, padding=1, bias=false)
		self.bn1   = nn.BatchNorm2d(out_channels)
		self.relu  = nn.ReLU(inplace = True)
		
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample
	
	def forward(self, x):
		identity = x
		
		#Main path của Residual Block -----------
		out = self.conv1(x)  #Lớp tích chập thứ 1
		out = self.bn1(out)  #Chuẩn hóa Batch
		out = self.relu(out) #ReLU
			
		out = self.conv2(out)
		out = self.bn2(out)
		#----------------------------------------
			
		if self.downsample is not None: #Projection Shortcut
			identity = self.downsample(x)
			
		out += identity       #Cộng hàm phần dư với input (kết nối tắt): elements-wise
		out = self.relu(out)
		return out
```
- **Lớp tích chập thứ nhất**: `self.conv1 = nn.conv2d()`
	- `in_channels`: số kênh đầu vào của khối.
	- `out_channels`: số kênh đầu ra của khối.
	- `kernel_size=3`: kích thước kernel tích chập là $3\times 3$.
	- `stride=stride`: bước nhảy của kernel tích chập, quyết định việc giảm kích thước đặc trưng.
	- `padding=1`: thêm padding để giữ nguyên kích thước đặc trưng khi `stride=1`.
	- `bias=False`: không sử dụng bias vì đã có BatchNorm.
- **Chuẩn hóa Batch**: `self.bn1 = nn.BatchNorm2d(out_channels)`
	- `nn.BatchNorm2d`: lớp chuẩn hóa Batch cho dữ liệu 2 chiều (ảnh).
	- `out_channels`: số lượng features để chuẩn hóa (tương ứng với số kênh đầu ra của lớp Conv2d).
- **Hàm kích hoạt ReLU**: `self.relu = nn.ReLU(inplace=True)`
	- `nn.ReLU`: hàm kích hoạt ReLU (Rectified Linear Unit) áp dụng phi tuyến tính.
	- `inplace=True`: thực hiện phép toán ReLU trực tiếp trên input để tiết kiệm bộ nhớ.
- Lớp tích chập thứ 2:
	- Tương tự lớp thứ nhất nhưng với giá trị `stride = 1` mặc định.
- **Lớp downsample cho projection shortcut** (tích chập $1\times 1$) nếu cần: `self.downsample = downsample`
	- Tùy chọn, có thể là lớp Sequential chứa Conv2d và BatchNorm2d để thực hiện projection shortcut.
	- Được sử dụng khi `stride > 1` hoặc khi số kênh đầu vào và đầu ra khác nhau, *để kích thước identity và hàm phần dư khớp nhau khi cộng*.
#### 1.1.1 Giá trị `bias` trong hàm nn.Conv2d
- Trong lớp tích chập 2 chiều (nn.Conv2d) của PyTorch (cũng như trong các thư viện deep learning khác), *tham số bias là một vector các giá trị* có thể học được, được *cộng vào output* của phép tích chập.
- Về cơ bản, `bias` là một *tham số bổ sung* cho mỗi kernel tích chập, giúp *tăng thêm độ linh hoạt* cho mô hình.
- **Vai trò**:
	- Dịch chuyển Hàm kích hoạt - tăng khả năng biểu diễn:
		- Cho phép dịch chuyển `output` của phép tích chập trước khi đưa vào hàm kích hoạt.
		- Không có `bias`, thì chỉ có phép biến đổi tuyến tính (tích chập và tổng) và hàm kích hoạt phi tuyến luôn được áp dụng quanh điểm gốc (như ReLU luôn chặn giá trị âm về $0$).
		- `Bias` dịch chuyển "điểm gốc" này, cho phép hàm kích hoạt hoạt động trên các vùng không gian khác nhau của `input` từ đó tăng khả năng biểu diễn của mạng.
	- Tăng tính linh hoạt:
		- Tăng thêm một chiều tự do, cho phép học các hàm số phức tạp hơn.
		- Nó tương tự như hệ số tung độ gốc (intercept) trong phương trình đường thẳng $y=mx+c$, trong đó $c$ là bias.
		- Nghĩa là cho phép đường thẳng/siêu phẳng dịch chuyển, mở rộng phạm vi biểu diễn.
- Giá trị `bias = False` trong kiến trúc ResNet:
	- Do có lớp `BatchNorm2d` (Batch Normalization) đã bao gồm các tham số $\text{beta}$ (shift) và $\text{gamma}$ (scale) tương tự chức năng và thậm chí mạnh mẽ hơn trong kiểm soát phân phối.
	- Việc có cả 2 có thể gây chậm hội tụ, phức tạp mô hình.
#### 1.1.2 Batch Normalization
- `BatchNorm2d`: là *lớp chuẩn hóa activations* của một lớp trong từng mini-batch dữ liệu.
- **Mục đích**:
	- Tăng tốc huấn luyện: cho phép sử dụng $\text{learning rate}$ cao hơn, và khi `activations` được chuẩn hóa, gradient trở nên ổn định và mô hình học nhanh hơn.
	- Cải thiện độ ổn định:
		- Giảm hiện tượng **internal covariate shift** (sự thay đổi phân phối `activations` giữa các lớp trong khi huấn luyện).
		- Chuẩn hóa `activations`, các lớp sau nhận được input có phân phối ổn định hơn, làm cho quá trình huấn luyện ổn định hơn và ít bị ảnh hưởng bởi sự thay đổi phân phối input.
	- Hiệu Ứng Chính Quy Hóa (Regularization Effect):
		- Batch Normalization có một hiệu ứng chính quy hóa nhẹ, giảm overfitting và cải thiện khả năng tổng quát hóa.

- **Hoạt động**: bao gồm các bước
	1. **Tính toán Thống kê theo Mini-batch**: với mỗi batch, tính giá trị $\text{mean}$ (trung bình) $\mu_B$ và $\text{standard deviation}$ (độ lệch chuẩn) $\sigma_B$ của `activation` trên từng kênh (feature channel).
		- Input BatchNorm: $x$ có kích thước $(N, C, H, W)$: $N$ - batch size, $C$ - số kênh, $H$ - chiều cao, $W$ - chiều rộng. 
		- $\text{mean}$ và $\text{standard deviation}$ được tính cho mỗi kênh $c$ (từ 1 đến $C$) trên tất cả các mẫu trong batch và trên tất cả các vị trí không gian $(H, W)$.
		 - Công thức:
			 - $\begin{cases} \mu_B^{(c)} &= \dfrac{1}{N \times H \times W} \displaystyle\sum_{i=1}^{N} \sum_{j=1}^{H} \sum_{k=1}^{W} x_{ijk}^{(c)} \\[6pt] (\sigma_B^{(c)})^2 &= \dfrac{1}{N \times H \times W} \displaystyle\sum_{i=1}^{N} \sum_{j=1}^{H} \sum_{k=1}^{W} (x_{ijk}^{(c)} - \mu_B^{(c)})^2 \end{cases}$
		- Trong đó, $x_{ijk}^{(c)}$ là giá trị activation tại vị trí $(i, j, k)$ trên kênh $c$ trong batch $B$.
	2. **Chuẩn hóa (Normalize) Activation**: sử dụng hai giá trên
		- $\hat{x}_{ijk}^{(c)} = \dfrac{x_{ijk}^{(c)} - \mu_B^{(c)}}{\sqrt{(\sigma_B^{(c)})^2 + \varepsilon}}$
        - Trong đó, $\varepsilon$ là một số nhỏ (ví dụ, $10^{-5}$) để tránh chia cho $0$.
	3. **Scale và Shift (Tỷ Lệ và Dịch Chuyển)**:
		- Sau chuẩn hóa, *phép biến đổi scale và shift* áp dụng sử dụng hai tham số có thể học được: *gamma* $\gamma$ và *beta* $\beta$.
		- Các tham số này được học trong quá trình huấn luyện và cho phép mạng học lại phân phối `activations` tối ưu cho từng lớp.
			- $y_{ijk}^{(c)} = \gamma^{(c)} \hat{x}_{ijk}^{(c)} + \beta^{(c)}$
		- Trong đó, $\gamma^{(c)}$ và $\beta^{(c)}$ là tham số scale và shift cho kênh $c$, được khởi tạo lần lượt bằng 1 và 0.
	4. **Trong Quá Trình Đánh Giá (Evaluation/Inference)**:
		- Trong quá trình đánh giá hoặc inference, $\text{mean}$ và $\text{standard deviation}$ không được tính toán theo batch nữa, mà sử dụng **running mean** và **running variance** được tích lũy trong quá trình huấn luyện trên toàn bộ tập huấn luyện.
		- Điều này đảm bảo rằng output của BatchNorm là deterministic (xác định) và không phụ thuộc vào kích thước batch trong quá trình đánh giá. 
### 1.2 ResNet Module
1. **ResNet Module** - Basic Block (tương tự Residual Block nhưng cụ thể hóa hơn)
```python
class BasicBlock(nn.Module):
	expansion = 1 #Hệ số mở rộng kênh, mặc định với BasicBlock là 1.
	def __init__(self, inplanes, planes, stride=1, downsample=None)
		super(BasicBlock, self).__init__()
		#Lớp tích chập:
		# - inplanes: đầu vào
		# - planes: số kênh trung gian (đầu ra luôn)
		self.conv1 = nn.conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1   = nn.BatchNorm2d(planes)
		self.relu  = nn.ReLU(inplace=True)	
		
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2   = nn.BatchNorm2d(planes)
		
		self.downsample = downsample
		#Lưu lại giá trị stride (trong _make_layer)
		self.stride = stride
		
	def forward(self, x):
		identity = x
		
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		
		out = self.conv2(out)
		out = self.bn2(out)
		
		if self.downsample is not None:
			identity = self.downsample(x)
		
		out += identity
		out  = self.relu(out)
		
		return out
```
- BasicBlock về cơ bản giống ResidualBlock, nhưng được đặt tên và tham số hóa rõ ràng hơn để dễ dàng sử dụng trong kiến trúc ResNet.
- Giá trị `expansion = 1` nghĩa là không thay đổi số kênh đầu ra so với số kênh trung gian.
2. **Bottleneck Block**
```python
class Bottleneck(nn.Module):
	#Hệ số mở rộng kênh, bằng 4 cho Bottleneck
	# - expansion = 4: số kênh đầu ra gấp 4 lần số kênh trung gian (planes * expansion).
	# - Giúp giảm số lượng tham số, tính toán.
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False) #1x1 giảm
        self.bn1   = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False) #3x3 chính
        self.bn2   = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)          #1x1 tăng
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
		
    def forward(self, x):
        identity = x
		
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
		
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
		
        out = self.conv3(out)
        out = self.bn3(out)
		
        if self.downsample is not None:
            identity = self.downsample(x)
		
        out += identity
        out  = self.relu(out)
		
        return out
```
- Lớp tích chập $1\times 1$ với số kênh $\textcolor{lightblue}{\text{inplanes}}$, **lớp Bottleneck**: (giảm số lượng kênh)
	- Đầu vào có kích thước: $H\times W\times C_\text{in}$
	- Một kernel kích thước: $1\times 1\times C_\text{in}$, với $C_\text{in} = \textcolor{lightblue}{\text{inplanes}}$ và số lượng kernel là $\textcolor{lightgreen}{\text{planes}}$.
	- Khi đó, đầu ra có kích thước: $H\times W\times C_\text{out}$ với $C_\text{out} = \textcolor{lightgreen}{\text{planes}}$.
- Hoạt động giảm số kênh:
	- Tại mỗi pixel, thay vì lấy vùng lân cận (như $3 \times 3$), bộ lọc chỉ lấy tất cả các giá trị trên các kênh.
	- Mỗi bộ lọc thực hiện tích vô hướng (dot product) giữa vector đầu vào và vector trọng số.
	- Kết quả là một số vô hướng, tạo thành một kênh đầu ra.
	- Số lượng đầu ra bằng số lượng bộ lọc.
- Bottleneck sử dụng 3 lớp tích chập ($1\times 1, 3\times 3, 1\times 1$) để giảm độ phức tạp tính toán.
- Giá trị `expansion = 4` chỉ ra rằng số kênh đầu ra của khối bottleneck gấp 4 lần số kênh trung gian.
## 2. Kiến trúc ResNet hoàn chỉnh
```python
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
    #block: loại khối (Basic/Bottleneck), layers: số khối trong mỗi lớp
        super(ResNet, self).__init__()
        #Số kênh ảnh đầu vào
        self.inplanes = 64
        #Lớp tích chập đầu tiên (input channel=1 cho grayscale FashionMNIST)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        #Lớp gộp cực đại đầu tiên
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        ### RESNET MODULES
        self.layer1 = self._make_layer(block, 64, layers[0])            #ResNet 1
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #ResNet 2, stride=2 giảm kích thước
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #ResNet 3, stride=2 giảm kích thước
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #ResNet 4, stride=2 giảm kích thước
        
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                     #Lớp gộp trung bình toàn cục
        self.fc = nn.Linear(512 * block.expansion, num_classes)         #FC: Lớp kết nối đầy đủ
		
	#Hàm tạo ResNet Module
    def _make_layer(self, block, planes, blocks, stride=1): 
	    #Khởi tạo downsample cho projection shortcut
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #Lớp downsample (projection shortcut)
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
			
        layers = []
        #Khối phần dư đầu tiên
        layers.append(
	        block(self.inplanes, planes, stride, downsample)
	    )
		self.inplanes = planes * block.expansion                        #Cập nhật inplanes
        #Thêm các khối phần dư còn lại (không có downsample)
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
			
        return nn.Sequential(*layers)
		
    def forward(self, x):
        x = self.conv1(x)       #Lớp tích chập đầu tiên
        x = self.bn1(x)         #Chuẩn hóa Batch
        x = self.relu(x)        #ReLU
        x = self.maxpool(x)     #Gộp cực đại
		
		x = self.layer1(x)      #Lớp ResNet 1
        x = self.layer2(x)      #Lớp ResNet 2
        x = self.layer3(x)      #Lớp ResNet 3
        x = self.layer4(x)      #Lớp ResNet 4
		
        x = self.avgpool(x)     #Gộp trung bình toàn cục
        x = torch.flatten(x, 1) #Flatten đặc trưng
        x = self.fc(x)          #Lớp kết nối đầy đủ
		
        return x
```
- `__init__` khởi tạo lớp tích chập đầu tiên, lớp gộp cực đại, các lớp ResNet (mô-đun ResNet), lớp gộp trung bình toàn cục và lớp kết nối đầy đủ.
- `make_layer` là hàm helper để tạo ra một lớp ResNet (mô-đun ResNet) với số lượng khối phần dư và số kênh xác định.

- **Khởi tạo các biến thể ResNet**:
```python
def resnet18(num_classes, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)  #layers=[2, 2, 2, 2] cho ResNet-18
	
def resnet50(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)  #layers=[3, 4, 6, 3] cho ResNet-50
	
def resnet101(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs) #layers=[3, 4, 23, 3] cho ResNet-101
	
def resnet152(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, **kwargs) #layers=[3, 8, 36, 3] cho ResNet-152
```

# Tham khảo
```bibtex
@inproceedings{he2016deep,
    title ={Deep residual learning for image recognition},
    author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
    booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
    pages={770--778},
    year ={2016}
}

@article{DBLP:journals/corr/abs-2106-11342,
  author       = {Aston Zhang and
                  Zachary C. Lipton and
                  Mu Li and
                  Alexander J. Smola},
  title        = {Dive into Deep Learning},
  journal      = {CoRR},
  volume       = {abs/2106.11342},
  year         = {2021},
  url          = {https://arxiv.org/abs/2106.11342},
  eprinttype    = {arXiv},
  eprint       = {2106.11342},
  timestamp    = {Wed, 30 Jun 2021 16:14:10 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2106-11342.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
- https://d2l.aivivn.com/chapter_convolutional-modern/resnet_vn.html
- https://towardsdatascience.com/the-annotated-resnet-50-a6c536034758/