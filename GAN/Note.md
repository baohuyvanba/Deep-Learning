- **Nội dung**:
	- Sự khác biệt giữa mô hình tạo sinh (Generative) và mô hình phân biệt (Discriminative).
	- Xác định các vấn đề mà GAN có thể giải quyết.
	- Tìm hiểu vai trò của trình Generator và trình phân biệt trong hệ thống GAN.
	- Tìm hiểu ưu và nhược điểm của các hàm tổn thất GAN phổ biến.
	- Xác định các giải pháp có thể có cho các vấn đề thường gặp khi huấn luyện GAN.
	- Ứng dụng
# I. Introduction
- GAN thuộc nhóm Generative Model, là **mô hình tạo sinh**: tạo các thực thể dữ liệu mới giống với dữ liệu huấn luyện.
- GAN có thể tạo hình ảnh trông giống như ảnh chụp khuôn mặt người, dù các khuôn mặt không thuộc về bất kỳ người nào.
- Ví dụ với kết quả đầu ra của styleGAN:

![styleGAN](https://github.com/baohuyvanba/Deep-Learning/blob/main/GAN/attachments/styleGAN.png)
- Mạng đối nghịch tạo sinh - GAN là viết tắt cho Generative Adversarial Networks.
	- Generative giống như ở trên và Adversarial là đối nghịch.
	- GAN được cấu thành từ 2 mạng gọi là Generator và Discriminator, luôn đối nghịch đầu với nhau trong quá trình huấn luyện mạng GAN.

- Ứng dụng: tạo khuôn mặt người, thay đổi độ tuổi, nhân vật hoạt hình, Image-to-Image, Text-to-Image, Semantic Segmentation, khôi phục ảnh, ảnh siêu phân giải.
# II. Kiến trúc mạng GAN
## 1. GAN
- Gồm 2 mạng con là **Generator** và **Discriminator**.
	- Generator ($\mathcal{G}$): sinh ra các dữ liệu giống như thật;
	- Discriminator ($\mathcal{D}$): cố gắng phân biệt:
		- Dữ liệu thật: từ tập dữ liệu huấn luyện thực tế (ví dụ: ảnh khuôn mặt người thật).
		- Dữ liệu giả: tạo ra bởi Generator. Discriminator sẽ cố gắng đưa ra quyết định cho mỗi mẫu dữ liệu: đâu là dữ liệu thật và đâu là dữ liệu do Generator tạo ra.

- **Ý tưởng**: bắt nguồn từ [zero-sum non-cooperative game](https://cs.stanford.edu/people/eroberts/courses/soco/projects/1998-99/game-theory/nonzero.html).
	- Hiểu đơn giản như trò chơi đối kháng 2 người (cờ vua, cờ tướng), nếu một người thắng thì người còn lại sẽ thua. Ở mỗi lượt thì cả 2 đều muốn maximize cơ hội thắng của mình và minimize cơ hội thắng của đối phương.
	- Discriminator và Generator trong mạng GAN giống như 2 đối thủ trong trò chơi.
	- Trong lý thuyết trò chơi thì GAN model converge khi cả Generator và Discriminator đạt tới trạng thái Nash equilibrium, tức là 2 người chơi đạt trạng thái cân bằng và đi tiếp các bước không làm tăng cơ hội thắng. *"A strategy profile is a Nash equilibrium if no player can do better by unilaterally changing his or her strategy"*.

- Một cách chính thức hơn, với tập thực thể dữ liệu $X$ và tập các nhãn $Y$:
	- **Mạng tạo sinh** ($\mathcal{G}$) - Generative Model
		- Tạo ra mẫu dữ liệu mới.
		- Ghi lại xác suất chung $p(X, y)$ hoặc $P(X)$ nếu không nhãn ($P(X \vert y)$)
		- Dựa nhiều vào phân phối dữ liệu.
		- Gaussian Mixture Model, Naive Bayes Classifier.
	- **Mạng phân biệt** ($\mathcal{D}$) - Discriminative Model
		- Phân loại / dự đoán dựa trên các mẫu sẵn có - bản chất là một mô hình Classification.
		- Dự đoán xác suất có điều kiện $P(y \vert X)$.
		- Neural Network, Logistic Regression, SVM, Conditional Random Field, Decision Tree.
	- Cách phân chia này là một cách phân loại khác so với Supervised, Unsupervised, Semi-supervised.
- Mô hình tạo sinh bao gồm cả việc phân phối dữ liệu và cho bạn biết khả năng xảy ra của một ví dụ nhất định.

- **Mô hình hóa xác suất**:
	- Cả hai loại mô hình (tạo sinh, phân biệt) không trả về một số đại diện cho xác suất - có thể lập mô hình phân phối dữ liệu dựa theo quá trình phân phối đó.
	- Ví dụ:
		- Thuật toán phân loại phân biệt như [cây quyết định](http://wikipedia.org/wiki/Decision_tree_learning) có thể *gắn nhãn cho một thực thể mà không gán xác suất*.
		- Một bộ phân loại như vậy vẫn sẽ là một mô hình vì việc phân phối tất cả nhãn được dự đoán sẽ mô hình hoá việc phân phối thực tế của nhãn trong dữ liệu.
	- Tương tự, mô hình tạo sinh có thể mô hình hoá một phân phối bằng cách tạo dữ liệu "giả" thuyết phục trông giống như được lấy từ phân phối đó.

- **Mô hình tạo sinh rất khó**:
	- Mô hình tạo sinh giải quyết một nhiệm vụ khó khăn hơn so với các mô hình phân biệt tương tự.
	- Mô hình tạo sinh cho hình ảnh có thể ghi lại các mối tương quan như "những thứ trông giống như thuyền có thể sẽ xuất hiện gần những thứ trông giống như nước" và "mắt khó có thể xuất hiện trên trán". Đây là những bản phân phối rất phức tạp.
	- Ngược lại, mô hình phân biệt có thể tìm hiểu sự khác biệt giữa "tàu thuyền" hoặc "không phải tàu thuyền" chỉ bằng cách tìm một vài mẫu rõ ràng. Mô hình này có thể bỏ qua nhiều mối tương quan mà mô hình tạo sinh phải nắm bắt đúng.
	- Mô hình phân biệt cố gắng vẽ ranh giới trong không gian dữ liệu, trong khi mô hình tạo sinh cố gắng mô hình hoá cách dữ liệu được đặt trong không gian.
	- Ví dụ: sơ đồ sau đây cho thấy các mô hình phân biệt và tạo sinh của chữ số viết tay:
 ![Generative & Discriminative](https://github.com/baohuyvanba/Deep-Learning/blob/main/GAN/attachments/GenerativeDiscriminative.png)
	- Mô hình phân biệt cố gắng phân biệt giữa chữ số 0 và 1 viết tay bằng cách vẽ một đường trong không gian dữ liệu. Nếu nhận được đúng đường thẳng, mô hình này có thể phân biệt 0 với 1 mà không cần phải mô hình hoá chính xác vị trí các thực thể được đặt trong không gian dữ liệu ở một bên của đường thẳng.
	- Ngược lại, mô hình tạo sinh cố gắng tạo ra các số 1 và 0 thuyết phục bằng cách tạo các chữ số gần với các chữ số thực trong không gian dữ liệu. Mô hình này phải mô hình hoá quá trình phân phối trong không gian dữ liệu.
	- GAN cung cấp một cách hiệu quả để huấn luyện các mô hình phong phú như vậy sao cho giống với một phân phối thực tế. Để hiểu cách hoạt động của các mạng này, chúng ta cần hiểu cấu trúc cơ bản của GAN.
![](https://github.com/baohuyvanba/Deep-Learning/blob/main/GAN/attachments/GAN.png)
- Cả trình Generator và trình Discriminator đều là mạng nơron.
- Đầu ra của Generator được kết nối trực tiếp với đầu vào của Discriminator.
- Thông qua phương pháp lan truyền ngược.
## 2. Generator
### 2.1 Kiến trúc
- Generator là mô hình sinh ra dữ liệu, tức là với ví dụ ở trên là mạng sinh ra các chữ số giống với dữ liệu trong MNIST.
- Generator có input là noise (random vector) là output là chữ số.
- Generator học cách tạo dữ liệu giả bằng cách kết hợp phản hồi từ Discriminator. Mô hình này học cách làm cho phân loại đầu ra là dữ liệu thực.

- **Noise input**:
	- Các chữ số khi viết ra không hoàn toàn giống nhau.
		- Ví dụ số 0 ở hàng đầu tiên có rất nhiều biến dạng nhưng vẫn là số 0.
	- Input của Generator là noise để khi ta thay đổi noise ngẫu nhiên thì Generator có thể sinh ra một biến dạng khác của chữ viết tay.
	- Noise cho Generator thường được sinh ra từ normal distribution hoặc uniform distribution.
![MNIST](https://github.com/baohuyvanba/Deep-Learning/blob/main/GAN/attachments/MNIST.png)

- Khi đó, theo yêu cầu của bài toán, ta xác định mô hình mạng GAN với kiến trúc:
```python
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()    #Output values in the range [-1, 1]
        )
	    
    def forward(self, z):
        img = self.model(z)
        img = img.view(z.size(0), 1, 28, 28)
        return img
```
### 3.2 Huấn luyện
![GLoss](https://github.com/baohuyvanba/Deep-Learning/blob/main/GAN/attachments/GLoss.png)
- Quá trình huấn luyện Generator đòi hỏi mức độ tích hợp chặt chẽ hơn giữa Generator và Discriminator:
	- Dữ liệu đầu vào ngẫu nhiên, thường là nhiễu sinh ra từ normal distributation; 
	- Generator, biến đổi dữ liệu đầu vào ngẫu nhiên thành một thực thể dữ liệu;
	- Discriminator, phân loại dữ liệu được tạo;
	- Đầu ra của Discriminator;
	- Hàm mất mát của bộ tạo sinh $\to$ Generator.
- **Dữ liệu đầu vào ngẫu nhiên**:
	- Mạng nơron cần đầu vào, thông thường là một giá trị cần phân biệt, ... nhưng ở đây là mạng để tạo ra một thực thể dữ liệu mới.
	- Cơ bản nhất, GAN lấy nhiễu ngẫu nhiên làm đầu vào $\to$ đầu ra có ý nghĩa. Việc này cho phép GAN tạo và lấy mẫu từ nhiều vị trí trong phân phối mẫu mục tiêu.
	- Các thử nghiệm cho thấy rằng *việc phân phối nhiễu không quan trọng lắm*, vì vậy, chúng ta có thể chọn một giá trị dễ lấy mẫu, chẳng hạn như phân phối đồng nhất. Để thuận tiện, không gian lấy mẫu nhiễu thường có kích thước nhỏ hơn kích thước của không gian đầu ra.
- **Sử dụng Discriminator để huấn luyện Generator**:
	- *Huấn luyện mạng nơron*: thay đổi trọng số của mạng để giảm lỗi / mất mát đầu ra.
	- Tuy nhiên, trong GAN, $\mathcal{G}$ *không được kết nối trực tiếp với mất mát* mà ta đang cố gắng tác động.
	- Đầu ra của $\mathcal{G}$ sẽ đưa vào $\mathcal{D}$ từ đó sẽ tạo ra đầu ra mà chúng ta đang cố gắng tác động.
- Ta phải *đưa phần mạng bổ sung* này vào **quá trình lan truyền ngược**.
	- Phương pháp lan truyền ngược điều chỉnh từng trọng số theo đúng hướng bằng cách tính toán mức tác động của trọng số đối với đầu ra – mức độ thay đổi của đầu ra nếu thay đổi trọng số.
	- Tuy nhiên, *tác động của trọng số của* $\mathcal{G}$ *phụ thuộc vào tác động của trọng số của $\mathcal{D}$* được đưa vào.
	- Vì vậy, quá trình truyền ngược bắt đầu từ: Đầu ra $\to \mathcal{D} \to \mathcal{G}$.
	- Và đồng thời, $\mathcal{D}$ sẽ không thay đổi trong quá trình này, vì đơn giản nếu nhắm vào một mục tiêu di chuyển liên tục sẽ rất khó khăn.
	- **Quy trình**:
		1. Lấy mẫu nhiễu ngẫu nhiên;
		2. Tạo đầu ra của $\mathcal{G}$;
		3. Lấy đầu ra "thực" hay "giả" từ $\mathcal{D}$;
		4. Tính toán mất mát từ việc phân loại đối tượng phân biệt;
		5. Lan truyền ngược thông qua cả $\mathcal{D}$ và $\mathcal{G}$ để lấy độ dốc (gradient).
		6. Thay đổi trọng số của $\mathcal{G}$.
## 3. Discriminator
### 3.1 Kiến trúc
- Giá trị phân biệt trong GAN chỉ đơn giản là một bộ phân loại: phân biệt dữ liệu thực với dữ liệu do Generator tạo ra.
- Mô hình này có thể sử dụng bất kỳ cấu trúc mạng nào phù hợp với loại dữ liệu mà nó đang phân loại.
- Discriminator là mạng để phân biệt xem dữ liệu là thật (dữ liệu từ dataset) hay giả (dữ liệu sinh ra từ Generator).

- Trong bài toán với tập MNIST, discriminator dùng để phân biệt chữ số từ bộ MNIST và dữ liệu sinh ra từ Generator. Discriminator có input là ảnh biểu diễn bằng 784 chiều, output là kết quả ảnh thật hay ảnh giả.
- Đây đơn giản là bài toán Binary Classification.

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
			      nn.Sigmoid() #Output a probability
        )
	    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```
- **Discriminator**: Nhận ảnh (được flatten thành vector) và thông qua các lớp Fully Connected với hàm LeakyReLU, cuối cùng sử dụng Sigmoid để cho ra xác suất ảnh là thật hay giả.
### 3.2 Huấn luyện
- **Dữ liệu huấn luyện** của hàm phân biệt đến từ hai nguồn:
	- Các thực thể thực từ dữ liệu thực, chẳng hạn như ảnh chân dung người thật.
	- Các thực thể giả do Generator tạo.
- Trong quá trình huấn luyện Discriminator, Generator không được huấn luyện. Các trọng số của hàm này không đổi trong khi tạo các ví dụ để hàm phân biệt huấn luyện.

- Discriminator kết nối với hai **hàm mất mát**:
	- Discriminator loss (Mất mát của bộ phân biệt)
	- Generator loss (Mất mát của bộ tạo sinh)
- Trong quá trình huấn luyện Discriminator, bỏ qua mất mát của Generator.

- Trong **quá trình huấn luyện Discriminator**:
	1. Discriminator phân loại cả dữ liệu thực và dữ liệu giả từ Generator.
	2. Hàm mất mát sẽ phạt Discriminator vì phân loại sai.
	3. Discriminator cập nhật trọng số thông qua phương pháp lan truyền ngược.
## 4. Training
- GAN bao gồm hai mạng được huấn luyện ($\mathcal{D}$ và $\mathcal{G}$), do đó, thuật toán huấn luyện phải giải quyết hai vấn đề:
	1. Phải thực hiện huấn luyện trên hai quá trình khác biệt;
	2. Sự hội tụ của GAN khó xác định.
- **Đào tạo xen kẽ**: GAN huấn luyện hai bộ $\mathcal{D}$ và $\mathcal{G}$ xen kẽ với nhau
	- $\mathcal{D}$ huấn luyện trên 1 hay nhiều epochs;
	- $\mathcal{G}$ tương tự;
	- Lặp lại hai quá trình trên.
- Các tham số của $\mathcal{G}$ được giữ nguyên trong quá trình huấn luyện $\mathcal{D}$: phân biệt dữ liệu thật & giả / học cách nhận ra các khuyết điểm của $\mathcal{G}$.
- Ngược lại, tương tự cho quá trình huấn luyện $\mathcal{G}$. Việc giữ nguyên tham số đã được giải thích ở trên.

- **Hội tụ (Convergence)**:
	- Việc $\mathcal{G}$ được cải thiện trong quá trình huấn luyện $\to$ $\mathcal{D}$ gặp khó khăn trong việc phân biệt. Nếu $\mathcal{G}$ đạt được mức độ hoàn hảo, $\mathcal{D}$ sẽ chỉ đưa ra kết quả một cách ngẫu nhiên như việc tung đồng xu :v
	- Tuy nhiên, việc này đặt ra một vấn đề: đầu ra của $\mathcal{D}$ sẽ dần trở nên vô nghĩa khi số vòng lặp tăng lên $\to$ nếu GAN tiếp tục ghi nhận và huấn luyện theo đầu ra lúc này, nó sẽ chỉ đi theo kết quả một cách sai lầm.
	- Vì vậy, với GAN, *sự hội tụ không thể xác định một cách chắc chắn* lâu dài mà chỉ chớp nhoáng.
## 5. Loss function
- GAN cố gắng tái tạo phân bố xác suất $\to$ *hàm mất mát phản ánh sự khác nhau của phân bố đầu ra và phân bố thực*.
- Nhiều cách tiếp cận đã được đưa ra, trong đó, phổ biến:
	- **Minimax loss**: được đưa ra trong nghiên cứu GANs - [paper](https://arxiv.org/abs/1406.2661);
	- **Wasserstein loss**: TF-GAN - [paper](https://arxiv.org/abs/1701.07875).
- GANs có thể có tới hai hàm mất mát, một cho $\mathcal{D}$ và một cho $\mathcal{G}$ tác động đồng thời và nhằm phản ảnh sự khác nhau trong phân bố dữ liệu.

- Các phương pháp tính toán hàm mất mát ta thảo luận sẽ dựa trên việc đo lường khoảng cách giữa các phân phối xác suất.
- Dù vậy, $\mathcal{G}$ chỉ có thể điều chỉnh một phần của phép đo này, đó là phần liên quan đến dữ liệu giả. Vì vậy, khi huấn luyện ta loại bỏ phần liên quan đến dữ liệu thật.
- Mặc dù cuối cùng cả hai hàm mất mát trông có vẻ khác nhau, nhưng cả hai đều cùng xuất phát từ một công thức.
### 5.1 Minimax Loss
- Được đề cập trong bài báo GANs, $\mathcal{G}$ sẽ tối thiểu hóa trong khi $\mathcal{D}$ tối đa hóa hàm mất mát sau: $$E_x[\log(\textcolor{red}{\mathcal{D}}(x))] + E_z[\log(1 - \textcolor{red}{\mathcal{D}}(\textcolor{green}{\mathcal{G}}(z)))]$$trong đó:
	- $\textcolor{red}{\mathcal{D}}(x)$ xác suất đầu ra của $\mathcal{D}$ cho dữ liệu thực $x$ là thực.
	- $E_x$ là giá trị mong muốn trên toàn bộ dữ liệu thực.
	- $\textcolor{green}{\mathcal{G}}(z)$ đầu ra của $\mathcal{G}$ với nhiễu đầu vào $z$ $\to$ $\textcolor{red}{\mathcal{D}}(\textcolor{green}{\mathcal{G}}(z)))$ xác suất là dữ liệu thực của đầu ra $\mathcal{G}$ tạo ra bởi $\mathcal{D}$.
	- $E_z$ là giá trị mong muốn trên toàn bộ dữ liệu giả.

- $\mathcal{G}$ đương nhiên không thể tác động tới $E_x[\log(\textcolor{red}{\mathcal{D}}(x))]$ $\to$ việc tối thiểu hóa mất mát tương đương tối thiểu hóa biểu thức: $E_z[\log(1 - \textcolor{red}{\mathcal{D}}(\textcolor{green}{\mathcal{G}}(z)))]$
- Công thức bắt nguồn từ giá trị Cross-Entropy từ phân phối dữ liệu thực và dữ liệu giả được tạo ra.

> [!NOTE] Cross-Entropy
> Cross-entropy giữa **phân phối xác suất thực tế** $t = (t_1, ..., t_C)$ và **phân phối xác suất dự đoán** $p = (p_1, ..., p_C)$ được định nghĩa là: $$CE(p,t) = -\sum_{i=1}^{C}{t_i\log p_i} > 0$$
> Chi tiết: [[../../Machine Learning Fundamentals/Session 9. Logistic Regression#1. Xây dựng hàm mất mát|Session 9. Logistic Regression]]
### 5.2 Modified Minimax Loss
- Như được đề cập trong bài viết, việc huấn luyện có thể mắc kẹt ở giai đoạn đầu khi mà công việc của $\mathcal{D}$ rất đơn giản.
- Do đó, ta được đề xuất việc tối đa hóa giá trị $\textcolor{red}{\mathcal{D}}(\textcolor{green}{\mathcal{G}}(z))$.
- Cài đặt: [github](https://github.com/tensorflow/tensorflow/blob/2007e1ba474030fcce840b0b8a599558e7d5998f/tensorflow/contrib/gan/python/losses/python/losses_impl.py)
### 5.3 Wasserstein Loss
- Source: [Google Dev](https://developers.google.com/machine-learning/gan/loss?hl=en#wasserstein-loss)
# III. GAN's problems
## 1. Vanishing Gradient
- Tiêu biến Gradient: các nghiên cứu chỉ ra rằng, nếu $\mathcal{D}$ hoạt động quá tốt $\to$ việc huấn luyện $\mathcal{G}$ sẽ thất bại (không cung cấp đủ thông tin cho việc huấn luyện)
- Khắc phục:
	- Modified Minimax.
	- Wasserstein Loss: được thiết kế để ngăn chặn việc này dù $\mathcal{D}$ đạt mức tối ưu.
## 2. Mode Collapse
- Diễn ra khi $\mathcal{G}$ chỉ tạo ra duy nhất một đầu ra / một tập hợp nhỏ đầu ra (với bất kể đầu vào) khi nó thấy đầu ra này đặc biệt hợp lí cho $\mathcal{D}$ - trên thực tế $\mathcal{G}$ luôn tìm đầu ra duy nhất hợp lí cho $\mathcal{D}$.
- $\mathcal{D}$ sẽ mắc bẫy trong tập đầu ra đó (luôn đưa ra xác suất dữ liệu giả) $\to$ $\mathcal{G}$ sẽ dễ dàng để thích nghi với $\mathcal{D}$ lúc đó.
- Khắc phục: buộc $\mathcal{G}$ mở rộng phạm vi - ngăn chặn tối ưu hóa cho một $\mathcal{D}$ duy nhất
	- Wasserstein Loss: cho phép huấn luyện $\mathcal{D}$ tối ưu mà không lo bị vấn đề tiêu biến Gradient (đề cập ở trên) $\to$ $\mathcal{G}$ luôn phải tìm một đầu ra mới tốt hơn.
	- GANs Unrolled: kết hợp phân loại của $\mathcal{D}$ hiện tại và các phiên bản $\mathcal{D}$ tiếp theo.
## 3. Failure to Converge
- Không hội tụ.
- Khắc phục:
	- Thêm nhiễu vào đầu vào của $\mathcal{D}$.
	- Xử lí trọng số của $\mathcal{D}$.
****

# Tham khảo
```bibtex
@article{goodfellow2014generative,
  title={Generative adversarial nets},
  author={Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
  journal={Advances in neural information processing systems},
  volume={27},
  year={2014}
}
```
- https://developers.google.com/machine-learning/gan
- https://phamdinhkhanh.github.io/2020/07/13/GAN.html#11-gan-v%C3%A0-c%C3%A1c-%E1%BB%A9ng-d%E1%BB%A5ng
- https://nttuan8.com/bai-1-gioi-thieu-ve-gan/
- https://nttuan8.com/gioi-thieu-series-gan-generative-adversarial-networks/
