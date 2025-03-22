# I. Introduction
- LangChain: **framework mã nguồn mở** đơn giản hóa việc xây dựng ứng dụng sử dụng mô hình ngôn ngữ lớn (LLMs).
- Mục tiêu chính của LangChain là **giảm độ phức tạp** trong việc xây dựng ứng dụng LLM, giúp nhà phát triển tập trung vào logic ứng dụng thay vì chi tiết kỹ thuật của LLM.
****
- Được viết bằng Python và JavaScript, LangChain cung cấp một bộ công cụ, thành phần và tích hợp toàn diện giúp:
	- **Tương tác với nhiều LLMs khác nhau**:
		- Cung cấp *giao diện thống nhất* để làm việc với các LLMs khác nhau;
		- *Dễ dàng chuyển đổi* giữa các mô hình mà không cần thay đổi code ứng dụng quá nhiều.
	- **Xây dựng chuỗi (chains) và tác nhân (agents) phức tạp**:
		- *Kết hợp các thành phần* khác nhau: prompts, LLMs, tools và memory $\to$ quy trình làm việc phức tạp và tự động;
		- Chains: *tạo ra các ứng dụng đa bước*, agents tự quyết hành động dựa trên input của người dùng và các công cụ.
	- **Quản lý bộ nhớ (memory) và lưu trữ dữ liệu**:
		- Cung cấp các module bộ nhớ để *duy trì trạng thái hội thoại và ngữ cảnh* qua nhiều lượt tương tác;
		- Hỗ trợ *tích hợp với các hệ thống lưu trữ* dữ liệu để đảm bảo tính liên tục của ứng dụng.
	- **Kết nối với các nguồn dữ liệu bên ngoài**:
		- Truy cập và sử dụng dữ liệu từ nhiều nguồn khác nhau: databases, APIs, websites, và document stores;
		- Xây dựng các ứng dụng *RAG* (Retrieval-Augmented Generation) và ứng dụng cần thông tin cập nhật hoặc chuyên biệt.
## 2. Sự phát triển và phổ biến của Langchain
- LangChain đã nhanh chóng trở thành một framework phổ biến và quan trọng trong cộng đồng AI kể từ khi ra mắt vào tháng 10 năm 2022 bởi Harrison Chase:
	- Dự án LangChain đã nhận được số lượng sao lớn trên GitHub, cho thấy sự quan tâm và ủng hộ mạnh mẽ;
	- LangChain có một cộng đồng nhà phát triển lớn và tích cực, đóng góp vào sự phát triển;
	- Được sử dụng bởi nhiều công ty lớn như LinkedIn và Uber, cũng như trong nhiều dự án và ứng dụng AI khác nhau;
    - Được cập nhật thường xuyên với các tính năng mới, cải tiến và sửa lỗi, cho thấy sự phát triển liên tục.
## 3. Tại sao sử dụng LangChain?
- Đơn giản hóa quá trình phát triển:
	- Các cung cụ và abstraction giảm bớt sự phức tạp khi làm việc với LLMs và APIs;
- Tăng tốc phát triển:
	- Nhờ các thành phần dựng sẵn, mẫu và tích hợp sẵn;
- Tăng tính linh hoạt và tùy biến:
	- Kiến trúc Modular cho phép tùy chỉnh và kết hợp;
- Dễ mở rộng và bảo trì:
	- Nhờ cấu trúc Modular;
- Giải quyết vấn đề cốt lõi khi làm việc với LLMs:
	- Quản lý prompt;
	- Kết nối dữ liệu;
	- Duy trì ngữ cảnh;
	- Xây dựng agent.
## 4. Ưu điểm Langchain
- **Kiến trúc Modular**:
	- Phân tác các tác vụ phức tạp $\to$ các thành phần tái sử dụng, khả năng mở rộng và bảo trì.
- **Quản lý prompt hiệu quả**:
	- Cung cấp các công cụ để thiết kế và tối ưu hóa prompt;
	- Đảm bảo đầu ra nhất quán thông qua templates và kĩ thuật few-shot learning.
- **Khả năng tích hợp đa dạng**:
	- Hỗ trợ nhiều LLMs, công cụ và APIs;
	- Cho phép xây dựng ứng dụng context-aware.
- **Cung cấp công cụ nâng cao**:
	- LangSmith: debugging, testing, monitoring;
	- LangServe, LangGraph Cloud để triển khai ứng dụng.
- Hỗ trợ Python và Java Script.
- Cộng đồng lớn mạnh, tài nguyên lớn.
# II. Các thành phần cốt lõi của LangChain
- Được xây dựng dựa trên một số thành phần cốt lõi, hoạt động cùng nhau để tạo nên các ứng dụng LLM mạnh mẽ:
	- (1) Model I/O; (2) Data Connection; (3) Chains; (4) Memory; (5) Agents.
## 1. Model I/O
- Input/Output của mô hình, là nhóm các thành phần chịu trách nhiệm tương tác với LLMs, bao gồm:
### 1.1 LLMs
- **Mô hình ngôn ngữ lớn**, đóng vai trò là bộ não của ứng dụng.
- LangChain hỗ trợ tích hợp nhiều mô hình LLMs khác nhau: GPT-3, GPT-3.5, GPT-4, LLaMA, ...
  
```python
# OpenAI GPT
from langchain_openai import OpenAI
llm = OpenAI(openai_api_key="...")
# Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### 1.2 Chat Models
- Là **ứng dụng LLMs được tối ưu cho các ứng dụng chatbot**.
- Được thiết kế để xử lí các đoạn hội thoại nhiều lượt và có khả năng duy trì ngữ cảnh tốt hơn các LLMs thông thường.
- LangChain *cung cấp các classes để làm việc với Chat Models*, cho phép bạn tạo ra các tin nhắn (messages) thuộc các vai trò khác nhau trong hội thoại, ví dụ:
	- `AIMessage`: tin nhắn do mô hình tạo ra;
	- `HumanMessage`: tin nhắn do người dùng tạo ra;
	- `SystemMessage`: tin nhắn hệ thống, dùng để hướng dẫn về vai trò và hành vi mong muốn ở mô hình;
	- `ChatMessage`: Class chung, có thể điều chỉnh vai trò.

```python
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(temperature=0)

messages = [
    SystemMessage(
        content="You are a virtual assistant that can translate any text input to Vietnamese..."
    ),
    HumanMessage(content="I love you"),
]

response = chat(messages)
print(response)
#Output: AIMessage(content='English - Tôi yêu bạn.')
```

### 1.3 Prompt Templates
- **Cho phép tạo ra các prompts động và có cấu trúc**.
- Thay vì prompt cứng nhắc, có thể tạo ra `templates` chứa các biến `input variables` mà giá trị sẽ được điền khi sử dụng.
	- Tạo ra prompt linh hoạt, tái sử dụng được cho nhiều mục đích khác nhau.
- Class `PromptTemplate`: được dùng để tạo mẫu prompt.

```python
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    template        = "Tell me a joke in {language}",
    input_variables = ["language"]
)

formatted_prompt = prompt.format(language="spanish")
print(formatted_prompt)
# Output: 'Tell me a joke in spanish'
```
- Có thể lưu và đọc prompt template dưới dạng JSON.

### 1.4 Output Parse
- **Bộ phân tích đầu ra**: xử lí kiến trúc đầu ra LLMs.
	- Đầu ra LLMs thường là `text`.
	- Đầu ra mong muốn có thể là `JSON`, `csv`, hoặc một định dạng nào khác.
	- Output Parse giúp phân tích cú pháp text đầu ra và chuyển đổi đầu ra thành định dạng mong muốn.

```python
from langchain.output_parsers import RegexParser

parser = RegexParser(regex = r"Product Description:\s*(.*)")
output = parser.parse(response)
print(output)
```

## 2. Data Connection (Kết nối dữ liệu/tài liệu)
- Là **nhóm các thành phần giúp LangChain kết nối và làm việc với dữ liệu bên ngoài**.
- Rất quan trọng để xây dựng các ứng dụng như RAG, và các ứng dụng cần truy cập thông tin từ các nguồn khác nhau.
### 2.1 Document Loaders
- Bộ tải tài liệu, cho phép đọc dữ liệu từ nhiều nguồn (`text`, `csv`, `PDF`, website, database và cách dịch vụ như Github, S3).
- Hỗ trợ nhiều loại khác nhau, giúp tích hợp dữ liệu từ nhiều nguồn khác nhau.

```python
from langchain_community.document_loaders import TextLoader

loader    = TextLoader("./main.rb")
documents = loader.load()
print(documents)
```

### 2.2 Text Spliting
- **Tách văn bản**: quá trình chia văn bản dài thành các đoạn nhỏ hơn, phù hợp để embedding và truy xuất.
- Các LLMs luôn có giới hạn độ dài input $\to$ chia thành văn bản thành các `chunk` để xử lí hiệu quả hơn.
- Các thuật toán tách văn bản:
	- `RecursiveCharacterTextSplitter`;
	- `LanguageTextSplitter` (tách theo ngôn ngữ lập trình).

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size    = 2000,
    chunk_overlap = 200
)
documents = splitter.split_documents(raw_documents)
```

### 2.3 Text Embedding
- **Mô hình Embedding văn bản**, chuyển văn bản $\to$ vector trong không gian đặc trưng.
- Các mô hình Embedding được hỗ trợ: OpenAI, Hugging Face, Cohere, Ollama.

```python
from langchain_community.embeddings import OllamaEmbeddings

embed_model = OllamaEmbeddings()
embeddings  = embed_model.embed_documents(documents)
```

### 2.4 Vector Stores
- Là các **database chuyên dụng để lưu trữ và tìm kiếm vector embedding** hiệu quả.
- Hỗ trợ tích hợp với nhiều Vector Stores phổ biến: Chroma, FAISS, Pinecone, Weaviate, ... :
	- Thực hiện tìm kiếm tương đồng (similarity search) nhanh chóng trên các vector embedding;
	- Truy xuất các đoạn văn bản liên quan đến truy vấn của người dùng.

```python
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(documents, OllamaEmbeddings())
```

### 2.5 Retrievers
- **Bộ truy xuất**, đây là Interface để truy vấn dữ liệu từ Vector Stores.
	- Đơn giản hóa việc tìm kiếm và truy xuất thông tin;
	- Cung cấp nhiều loại Retrievers khác nhau;
	- Hỗ trợ các chiến lược khác nhau.

```python
retriever = db.as_retriever(search_kwargs={"k": 1}) #top-k = 1
```

## 3. Chains
- Là **chuỗi các thành phần (Components) kết hợp với nhau $\to$ quy trình xử lí dữ liệu phức tạp**.
	- Cho phép kết hợp: LLMs, prompts, tools, memory và các thành phần khác;
	- Xây dựng ứng dụng đa bước, tự động hóa tác vụ và tạo workflows phức tạp.

### 3.1 Sequential Chains
- **Chuỗi thành phần tuần tự**, thực hiện các bước một cách tuyến tính:
	- Đầu ra bước trước $\to$ Đầu vào bước sau;
	- Phù hợp tác vụ đơn giản.
- Cung cấp:
	- `SimpleSequentialChain`: đơn giản, phù hợp chuỗi với 1 input và 1 output mỗi bước;
	- `SequentialChain`: phức tạp hơn, hỗ trợ nhiều input/ouput mỗi bước và cho phép đặt tên input/output keys.

```python
from langchain.chains import SimpleSequentialChain, LLMChain

chain  = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
result = chain.run("science")
print(result)
```

- **SimpleSequentialChain**:
	- `LLMChain`: chuỗi xử lý sử dụng một mô hình LLM để tạo ra văn bản dựa trên một prompt.
	- Thành phần cơ bản của `first_chain` và `second_chain` là `LLMChain`;
	- Chuỗi `chain` được tạo từ 2 thành phần chuỗi xử lí thực hiện theo thứ tự;
	- Tham số `verbose` yêu cầu in ra các bước trung gian và đầu ra từng chuỗi trong quá trình thực thi.

```python
from langchain.chains import SequentialChain, LLMChain

overall_chain = SequentialChain(
    chains = [
        LLMChain(... output_key = "title"),
        LLMChain(... output_key = "outline"),
    ],
    input_variables  = ["content", "style"],
    output_variables = ["title", "outline"],
    verbose = True,
)
video_outline = overall_chain({"content": "Deep Learning in 1 minutes", "style": "funny"})
print(video_outline)
```

- **SquentialChain**:
	- `LLMChain(... output_key = "title"`: một chuỗi xử lí trong chuỗi, và chỉ định đầu ra lưu trữ với khóa `output_key = "title"`.
	- `input_variables=["content", "style"]`:
		- Tham số này định nghĩa danh sách các biến đầu vào mà `SequentialChain` này mong đợi khi chạy;
		- Trong trường hợp này, nó cần một từ điển (dictionary) chứa các khóa `"content"` và `"style"`.
	- `output_variables=["title", "outline"]`:
		- Tham số này định nghĩa danh sách các biến đầu ra mà `SequentialChain` này sẽ tạo ra sau khi hoàn thành;
		- Các tên biến này tương ứng với các giá trị của `output_key` trong các `LLMChain` bên trong.

### 3.2 Custom Chains
- **Chuỗi tùy chỉnh**, để định nghĩa các quy trình xử lý dữ liệu phức tạp và tùy chỉnh.
- Có thể *tạo các class kế thừa từ Chain class* và định nghĩa các bước xử lý dữ liệu trong method `_call` hoặc `_acall`.
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

class CustomChain:
    def __init__(self, llm):
        self.llm   = llm
        self.steps = []
		
    def add_step(self, prompt_template):
        # ...
        chain = LLMChain(...)
        self.steps.append(chain)
		
    def execute(self, input_text):
        # ...
        for step in self.steps:
            input_text = step.run(input_text)
        return input_text
		
chain = CustomChain(llm)
chain.add_step("Summarize the following text...")
chain.add_step("Translate the following English text...")
result = chain.execute("LangChain is a powerful framework...")
print(result)
```

### 3.3 LLMChain
- Là loại **chain cơ bản và quan trọng nhất** trong LangChain, được sử dụng để kết hợp một `LLM` và một `PromptTemplate`.
```python
from langchain.chains import LLMChain

chain  = LLMChain(
			llm    = llm,
			prompt = prompt_template
		)
result = chain.run(content = "Deep Learning in 1 minutes", style = "funny")
print(result)
```

## 4. Memory
- Cho phép LangChain *duy trì trạng thái hội thoại và ngữ cảnh* qua nhiều lượt tương tác.
	- Rất quan trọng để xây dựng các ứng dụng chatbot và các ứng dụng cần duy trì ngữ cảnh hội thoại;
	- Cung cấp nhiều loại memory khác nhau, phù hợp cho các `use cases` khác nhau.

### 4.1 ConversationBufferMemory
- **Lưu trữ toàn bộ lịch sử hội thoại trong bộ nhớ đệm**.
- Phù hợp cho hội thoại ngắn.
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
			{"input": "Hi, I'm Alice"},
			{"output": "Hello Alice, how can I help you today?"}
		)
memory.save_context(
			{"input": "What's the weather like?"},
			{"output": "I'm sorry, I don't have real-time weather information."}
		)

print(memory.load_memory_variables({}))
```

### 4.2 ConversationSummaryMemory
- **Tóm tắt lịch sử hội thoại thay vì lưu trữ toàn bộ**.
- Phù hợp cho các hội thoại dài, cần duy trì ngữ cảnh nhưng không muốn lưu trữ quá nhiều thông tin chi tiết.
```python
from langchain.memory import ConversationSummaryMemory
from langchain.llms import Ollama

memory = ConversationSummaryMemory(llm = llm) #Tóm tắt sử dụng LLM

memory.save_context(
			{"input"  : "Hi, I'm Alice"},
			{"output" : "Hello Alice, how can I help you today?"}
		)
memory.save_context(
			{"input"  : "I'm looking for a good Italian restaurant"},
			{"output" : "Great! I'd be happy to help you find a good Italian restaurant."}
		)

print(memory.load_memory_variables({}))
```

### 4.3 Lưu trữ và truy xuất bộ nhớ
- LangChain **hỗ trợ lưu trữ bộ nhớ vào file hoặc database** để duy trì trạng thái hội thoại qua các phiên.
- Có thể tạo `custom memory` classes để lưu trữ bộ nhớ theo cách riêng: ví dụ JSON file, SQLite database.
```python
import json

class PersistentMemory:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_memory()
		
    def load_memory(self):
        ...
		
    def save_memory(self):
	    ...

memory = PersistentMemory(file_path = 'conversation_history.json')
```

## 5. Agents (Tác nhân)
- Là các **thực thể tự động có thể sử dụng tools và đưa ra quyết định** để hoàn thành các tác vụ.
	- Kết hợp LLMs với các external tools để giải quyết các vấn đề phức tạp và tự động hóa các quy trình làm việc;
	- Hữu ích để xây dựng các ứng dụng conversational AI và tự động hóa các workflows phức tạp.

### 5.1 Built-in Tools
- **Công cụ tích hợp sẵn** được LangChain cung cấp để:
	- `WikipediaQueryRun`: Tìm kiếm thông tin trên Wikipedia;
	- `SerpAPIWrapper`: Tìm kiếm thông tin trên web thông qua SerpAPI;
	- `LLM-math`: Thực hiện các phép toán;
	- `load_tools`: Load nhiều tools cùng lúc.
```python
from langchain_community.tools import WikipediaQueryRun

api_wrapper = WikipediaAPIWrapper(...)
tool        = WikipediaQueryRun(api_wrapper = api_wrapper)

print(tool.name)        # Output: wikipedia
print(tool.description) # Output: A wrapper around Wikipedia...
print(tool.args)        # Output: {'query': {'title': 'Query', 'type': 'string'}}

result = tool.run({"query": "LangChain"})
print(result)
```

```python
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

tools = load_tools(["wikipedia", "llm-math"], llm = llm)
agent = initialize_agent(
    tools,
    llm,
    agent   = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True,
)
result = agent.run("What is the square root of the year Plato was born?")
print(result)
```

### 5.2 Custom Tools
- Custom Tools cho phép agent thực hiện các tác vụ cụ thể theo yêu cầu của ứng dụng.
- Để tạo Custom Tool, cần định nghĩa một `function` và decorate nó với `@tool decorator`.
```python
from langchain.tools import tool

@tool("search")
def search(query: str) -> str:
    """
    Look up things online.
    """
    return "LangChain"

@tool("multiply")
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b

tools = [search, multiply]
agent = initialize_agent(
			tools,
			llm,
			...
		)

result = agent.invoke("Multiply two numbers: 2 and 3")
print(result)
# Output: {'input': 'Multiply two numbers: 2 and 3', 'output': 'Action: multiply...'}
```

### 5.3 Cấu trúc và hoạt động của LLMs
- Mỗi Agent có thể coi là một tập hợp của nhiều tools.
- User input $\to$ Agent quyết định sử dụng tool nào, input cho tool đó là gì $\to$ Trả lời câu hỏi / hoàn thành tác vụ.
- **Quá trình hoạt động** của Agent có thể được mô tả như sau:
	1. User input.
	2. Agent quyết định: tool nào + input cho tool.
	3. Tool: chạy $\to$ trả về output.
	4. Agent: nhận output $\to$ quyết định bước tiếp theo.
	5. Lặp lại quá trình trên $\to$ kết quả cuối cùng.

- LangChain **cung cấp nhiều loại Agent** khác nhau:
	- Zero-shot ReAct Agent:
		- Phổ biến nhất;
		- Sử dụng ReAct framework để quyết định hành động dựa trên mô tả của tools.
	- Structured Chat Zero Shot React Description Agent:
		- Cải tiến của Zero-shot ReAct Agent;
		- Hỗ trợ input cho tool có nhiều hơn 1 tham số.
	- Custom Agent:
		- Có thể tạo custom agent class để định nghĩa logic hoạt động riêng.

# III. Các thành phần và ứng dụng nâng cao
## 1. Prompt Engineer nâng cao
- Nhằm tối ưu hóa và cải thiện hiệu suất mô hình LLMs.

### 1.1 Few-shot Learning
 - Kỹ thuật **cung cấp cho mô hình một vài ví dụ** mẫu (few-shot examples): giúp mô hình hiểu rõ tác vụ
 - `FewShotPromptTemplate`
```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

fewshot_examples = [
    {"command": "Turn on the kitchen light", "action": "turn on", ...},
    {"command": "Turn off the TV in the living room", "action": "turn off", ...},
    {"command": "Increase the bedroom temperature by 2 degrees", "action": "Increase temperture", ...},
]

example_prompt  = PromptTemplate(...)
few_shot_prompt = FewShotPromptTemplate(
    examples       = fewshot_examples,
    example_prompt = example_prompt,
    prefix         = "Extract the detail information for an IoT input command...",
    suffix         = "Input command from user: {command}...",
    input_variables   = ["command"],
    example_separator = "\n\n",
)

formatted_prompt = few_shot_prompt.format(command="Turn off the bath room light")
print(formatted_prompt)
```
### 1.2 Kiểm soát phong cách và Cấu trúc đầu ra
- Cho phép **kiểm soát phong cách, cấu trúc, và độ dài** của đầu ra: cung cấp examples hoặc templates trong prompts.
- Cách định dạng mong muốn: `bullet list`, `JSON`, hoặc văn phong cụ thể.
```python
video_outline_prompt_template = PromptTemplate(
    input_variables = ["title"],
    template        = """Write a outline of a Youtube video about {title}. Output in the bullet list format."""
)
```

### 1.3 Quản lý Prompt và Response
- Cung cấp các methods để gửi prompts đến LLMs và xử lý responses hiệu quả.
- Có thể sử dụng `chain.run()` hoặc `agent.run()` để gửi prompt và nhận response.
- Cung cấp các utilities để parsing và processing LLM output: trích xuất thông tin liên quan cho các downstream tasks.
```python
from langchain.output_parsers import RegexParser

parser = RegexParser(regex = r"Product Description:\s*(.*)")
output = parser.parse(response)
print(output)
```

## 2. Tích hợp LLMs và các nguồn dữ liệu
- Hỗ trợ tích hợp với nhiều LLMs và nguồn dữ liệu khác nhau, giúp xây dựng các ứng dụng đa dạng và mạnh mẽ.

### 2.1 Tích hợp với Hugging Face
- Tích hợp tốt với Hugging Face Hub/Transformers: sử dụng các mô hình LLMs và embedding models từ Hugging Face.
- Dễ dàng load các mô hình Hugging Face vào LangChain và sử dụng chúng trong chains và agents.
```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id)

llm       = HuggingFacePipeline(
				model     = model,
				tokenizer = tokenizer
			)
```

### 2.2. Tích hợp với Vector Databases
[[Langchain LLM Framework#2. Data Connection (Kết nối dữ liệu/tài liệu)]]

### 2.3. Tích hợp với APIs
- Cho phép tích hợp với các APIs bên ngoài, giúp LLMs tương tác với các hệ thống và dịch vụ khác:
	- Mở ra nhiều khả năng ứng dụng, ví dụ, truy xuất dữ liệu real-time, thực hiện các phép tính toán phức tạp, gửi email, đặt lịch hẹn, ...
	- Custom Tools là cách chính để tích hợp APIs vào LangChain Agents.

```python
@tool("post-recommendation")
def post_recommendation(user_id: str) -> dict:
    """You call this function when user want to provide the recommended posts related for user."""
    response = requests.get("https://langchain-demo.free.mockoapp.net/post-recommendation")
    return response.json()
```
## 3. Customization và Fine-tuning LLMs
- LangChain cho phép customization và fine-tuning LLMs để cải thiện hiệu suất cho các tasks hoặc domains cụ thể.
- Fine-tuning là quá trình huấn luyện tiếp mô hình LLM trên một dataset nhỏ hơn, liên quan đến use case cụ thể:
	- Giúp mô hình thích ứng với các patterns ngôn ngữ và kiến thức chuyên biệt cần thiết;
	- Hỗ trợ fine-tuning các mô hình Hugging Face Transformers.
```python
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Fine-tuning dataset
dataset = load_dataset(...)
# Định nghĩa các tham số training
training_args = TrainingArguments(...) 

model_id  = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id)

# Fine-tune
trainer = Trainer(
			model = model,
			args  = training_args,
			train_dataset = dataset['train'],
			eval_dataset  = dataset['test']
		)
trainer.train()

llm = HuggingFacePipeline(model = model, tokenizer = tokenizer)
```

## 4. LangGraph và LangServe cho triển khai ứng dụng
- **LangGraph**:
	- Là orchestration framework cho production-ready apps, với các tính năng như persistence và streaming, lý tưởng cho agentic systems;
	- Mở rộng khả năng của LangChain bằng cách hỗ trợ xây dựng các ứng dụng stateful, multi-actor với LLMs;
	- LangGraph ecosystem:
		- Platform cho deployment;
		- Server cho API exposure;
		- SDKs, CLI và Studio environment cho UI development và debugging.

- **LangServe**:
	- Là dedicated library để expose LangChain chains như RESTful APIs;
	- Tích hợp chains vào các hệ thống hiện có, tạo điều kiện tương thích và khả năng mở rộng trên nhiều môi trường.

# IV. So sánh LangChain
## 1. Hugging Face Transformers
- **Điểm mạnh**:
	- Repository lớn các pre-trained models;
	- Linh hoạt cho fine-tuning và task-specific customization;
	- Cộng đồng lớn mạnh đóng góp models, datasets và tutorials.
- **So sánh với LangChain**:
	- Hugging Face Transformers tập trung vào cung cấp models và công cụ fine-tuning;
	- LangChain tập trung vào xây dựng ứng dụng LLM với các components, chains, agents, memory và integrations;
	- Tích hợp tốt với Hugging Face Transformers để sử dụng các models của Hugging Face.
## 2. Haystack

- **Điểm mạnh**:
	- Mạnh mẽ trong RAG, tập trung vào document retrieval và integration;
	- Tích hợp tốt với search tools như Elasticsearch và vector databases;
	- Pre-built pipelines đơn giản hóa việc phát triển question-answering systems.
- **So sánh với LangChain**:
	- Haystack chuyên biệt cho search và question-answering systems;
	- LangChain rộng hơn, hỗ trợ nhiều loại ứng dụng LLM khác nhau;
	- LangChain cũng có RAG capabilities mạnh mẽ và tích hợp tốt với vector databases.
## 3. Flowise

- **Điểm mạnh của Flowise**:
	- No-code LLM app development;
	- Drag-and-drop interface, beginner-friendly.
- **So sánh với LangChain**:
	- Flowise hướng đến người dùng không có kinh nghiệm code;
	- LangChain hướng đến nhà phát triển chuyên nghiệp cần sự linh hoạt và khả năng tùy biến cao;
	- Flowise dễ sử dụng hơn cho các tác vụ đơn giản, nhưng LangChain mạnh mẽ hơn cho các ứng dụng phức tạp.
## 4. AutoChain

- **Điểm mạnh của AutoChain**:
	- Lightweight agent customization;
	- Đơn giản hơn cho agent workflows, dễ debug.
- **So sánh với LangChain**:
	- AutoChain tập trung vào agent customization;
	- LangChain cung cấp hệ sinh thái toàn diện hơn với nhiều components, chains, agents, memory và integrations.
	- LangChain có feature set toàn diện hơn, nhưng AutoChain có thể nhẹ hơn và đơn giản hơn cho một số use cases agent-centric.

- LangChain thường được ưu tiên cho phát triển ứng dụng LLM một cách toàn diện nhờ tích hợp rộng rãi và cộng đồng hỗ trợ mạnh mẽ.

# V. Sử dụng cơ bản
- **Cài đặt LangChain**
```bash
pip install langchain
npm install langchain
```
- **Thiết lập API Key**: cho các LLM providers (ví dụ, OpenAI API key) dưới dạng environment variables.
```bash
export OPENAI_API_KEY="your-api-key"
```
**Xây dựng ứng dụng LangChain đầu tiên**
```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm    = OpenAI(temperature = 0.9)
prompt = PromptTemplate(
			input_variables = ["text"],
			template = "Translate the following text from English to Spanish: {text}"
		)

chain = LLMChain(llm = llm, prompt = prompt)
translated_text = chain.run("Hello, world!")
print(translated_text)
```