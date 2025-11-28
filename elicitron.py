import operator
from typing import Annotated, Any, Optional

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from rich import print as rprint

# 從 .env 檔案讀取環境變數
load_dotenv()


# 代表人物設定（Persona）的資料模型
class Persona(BaseModel):
    name: str = Field(..., description="人物設定的名稱")
    background: str = Field(..., description="人物設定所具備的背景")


# 代表多個人物設定（Persona）的資料模型
class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="人物設定的清單"
    )


# 代表訪談內容的資料模型
class Interview(BaseModel):
    persona: Persona = Field(..., description="訪談對象的人物設定")
    question: str = Field(..., description="訪談中的提問")
    answer: str = Field(..., description="訪談中的回答")


# 代表訪談結果清單的資料模型
class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="訪談結果的清單"
    )


# 代表評估結果的資料模型
class EvaluationResult(BaseModel):
    reason: str = Field(..., description="判斷的理由")
    is_sufficient: bool = Field(..., description="資訊是否足夠")


# 需求定義生成式 AI 代理的狀態
class InterviewState(BaseModel):
    user_request: str = Field(..., description="使用者的請求")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="已生成的人物設定清單"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="已進行的訪談清單"
    )
    requirements_doc: str = Field(default="", description="已生成的需求定義文件")
    iteration: int = Field(
        default=0, description="人物設定生成與訪談的反覆次數"
    )
    is_information_sufficient: bool = Field(
        default=False, description="資訊是否足夠"
    )


# 生成人物設定（Persona）的類別
class PersonaGenerator:
    def __init__(self, llm: ChatOpenAI, k: int = 5):
        self.llm = llm.with_structured_output(Personas)
        self.k = k

    def run(self, user_request: str) -> Personas:
        # 定義提示詞範本
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是為使用者訪談建立多樣化人物設定的專家。",
                ),
                (
                    "human",
                    f"請針對以下使用者請求所需的訪談，生成 {self.k} 位多樣化的人物設定。\n\n"
                    "使用者請求: {user_request}\n\n"
                    "每個人物設定請包含姓名與簡要背景，並確保在年齡、性別、職業與技術專長方面具有多樣性。",
                ),
            ]
        )
        # 建立用於生成人物設定的 Chain
        chain = prompt | self.llm
        # 生成人物設定
        return chain.invoke({"user_request": user_request})


# 執行訪談的類別
class InterviewConductor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, personas: list[Persona]) -> InterviewResult:
        # 生成問題
        questions = self._generate_questions(
            user_request=user_request, personas=personas
        )
        # 生成回答
        answers = self._generate_answers(personas=personas, questions=questions)
        # 根據問題與回答的組合建立訪談清單
        interviews = self._create_interviews(
            personas=personas, questions=questions, answers=answers
        )
        for inter in interviews:
            rprint(inter.model_dump())
        # 回傳訪談結果
        return InterviewResult(interviews=interviews)

    def _generate_questions(
        self, user_request: str, personas: list[Persona]
    ) -> list[str]:
        # 定義用於生成問題的提示詞
        question_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是根據使用者需求生成適當問題的專家。",
                ),
                (
                    "human",
                    "請針對以下人物設定相關的使用者請求，生成一個問題。\n\n"
                    "使用者請求: {user_request}\n"
                    "人物設定: {persona_name} - {persona_background}\n\n"
                    "問題請具體明確，並設計為能從該人物設定的觀點中引出重要資訊。",
                ),
            ]
        )
        # 建立用於生成問題的 Chain
        question_chain = question_prompt | self.llm | StrOutputParser()

        # 為每個人物設定建立問題查詢
        question_queries = [
            {
                "user_request": user_request,
                "persona_name": persona.name,
                "persona_background": persona.background,
            }
            for persona in personas
        ]
        # 以批次方式生成問題
        return question_chain.batch(question_queries)

    def _generate_answers(
        self, personas: list[Persona], questions: list[str]
    ) -> list[str]:
        # 定義用於生成回答的提示詞
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你將以以下人物設定的身分進行回答: {persona_name} - {persona_background}",
                ),
                ("human", "問題: {question}"),
            ]
        )
        # 建立用於生成回答的 Chain
        answer_chain = answer_prompt | self.llm | StrOutputParser()

        # 為每個人物設定建立回答查詢
        answer_queries = [
            {
                "persona_name": persona.name,
                "persona_background": persona.background,
                "question": question,
            }
            for persona, question in zip(personas, questions)
        ]
        # 以批次方式生成回答
        return answer_chain.batch(answer_queries)

    def _create_interviews(
        self, personas: list[Persona], questions: list[str], answers: list[str]
    ) -> list[Interview]:
        # 為每個人物設定依據問題與回答的組合建立訪談物件
        return [
            Interview(persona=persona, question=question, answer=answer)
            for persona, question, answer in zip(personas, questions, answers)
        ]



# 評估資訊是否充足的類別
class InformationEvaluator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(EvaluationResult)

    # 依據使用者請求與訪談結果，評估資訊是否充足
    def run(self, user_request: str, interviews: list[Interview]) -> EvaluationResult:
        # 定義提示詞
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是負責評估是否有足夠資訊來撰寫完整需求文件的專家。",
                ),
                (
                    "human",
                    "請根據以下使用者請求與訪談結果，判斷是否已蒐集到足以撰寫完整需求文件的資訊。\n\n"
                    "使用者請求: {user_request}\n\n"
                    "訪談結果:\n{interview_results}",
                ),
            ]
        )
        # 建立用於評估資訊充足性的 Chain
        chain = prompt | self.llm
        # 回傳評估結果
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"人物設定: {i.persona.name} - {i.persona.background}\n"
                    f"問題: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )


# 生成需求文件的類別
class RequirementsDocumentGenerator:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, user_request: str, interviews: list[Interview]) -> str:
        # 定義提示詞
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是根據已蒐集的資訊來撰寫需求文件的專家。",
                ),
                (
                    "human",
                    "請根據以下使用者請求，以及來自多位人物設定的訪談結果，撰寫一份需求文件。\n\n"
                    "使用者請求: {user_request}\n\n"
                    "訪談結果:\n{interview_results}\n"
                    "需求文件請包含以下章節:\n"
                    "1. 專案概述\n"
                    "2. 主要功能\n"
                    "3. 非功能性需求\n"
                    "4. 限制條件\n"
                    "5. 目標使用者\n"
                    "6. 優先順序\n"
                    "7. 風險與因應對策\n\n"
                    "輸出請務必使用中文。\n\n需求文件:",
                ),
            ]
        )
        # 建立用於生成需求文件的 Chain
        chain = prompt | self.llm | StrOutputParser()
        # 生成需求文件
        return chain.invoke(
            {
                "user_request": user_request,
                "interview_results": "\n".join(
                    f"人物設定: {i.persona.name} - {i.persona.background}\n"
                    f"問題: {i.question}\n回答: {i.answer}\n"
                    for i in interviews
                ),
            }
        )


# 生成需求文件的 AI 代理類別
class DocumentationAgent:
    def __init__(self, llm: ChatOpenAI, k: Optional[int] = None):
        # 初始化各種產生器
        self.persona_generator = PersonaGenerator(llm=llm, k=k)
        self.interview_conductor = InterviewConductor(llm=llm)
        self.information_evaluator = InformationEvaluator(llm=llm)
        self.requirements_generator = RequirementsDocumentGenerator(llm=llm)

        # 建立圖（Graph）
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        # 初始化圖（Graph）
        workflow = StateGraph(InterviewState)

        # 新增各個節點
        workflow.add_node("generate_personas", self._generate_personas)
        workflow.add_node("conduct_interviews", self._conduct_interviews)
        workflow.add_node("evaluate_information", self._evaluate_information)
        workflow.add_node("generate_requirements", self._generate_requirements)

        # 設定入口節點
        workflow.set_entry_point("generate_personas")

        # 新增節點之間的邊
        workflow.add_edge("generate_personas", "conduct_interviews")
        workflow.add_edge("conduct_interviews", "evaluate_information")

        # 新增條件式邊
        workflow.add_conditional_edges(
            "evaluate_information",
            lambda state: not state.is_information_sufficient and state.iteration < 5,
            {True: "generate_personas", False: "generate_requirements"},
        )
        workflow.add_edge("generate_requirements", END)

        # 編譯圖（Graph）
        return workflow.compile()

    def _generate_personas(self, state: InterviewState) -> dict[str, Any]:
        # 生成人物設定
        new_personas: Personas = self.persona_generator.run(state.user_request)
        return {
            "personas": new_personas.personas,
            "iteration": state.iteration + 1,
        }

    def _conduct_interviews(self, state: InterviewState) -> dict[str, Any]:
        # 執行訪談
        new_interviews: InterviewResult = self.interview_conductor.run(
            state.user_request, state.personas[-5:]
        )
        return {"interviews": new_interviews.interviews}

    def _evaluate_information(self, state: InterviewState) -> dict[str, Any]:
        # 評估資訊
        evaluation_result: EvaluationResult = self.information_evaluator.run(
            state.user_request, state.interviews
        )
        print(evaluation_result.is_sufficient)
        print(evaluation_result.reason)
        return {
            "is_information_sufficient": evaluation_result.is_sufficient,
            "evaluation_reason": evaluation_result.reason,
        }

    def _generate_requirements(self, state: InterviewState) -> dict[str, Any]:
        # 生成需求文件
        requirements_doc: str = self.requirements_generator.run(
            state.user_request, state.interviews
        )
        return {"requirements_doc": requirements_doc}

    def run(self, user_request: str) -> str:
        # 設定初始狀態
        initial_state = InterviewState(user_request=user_request)
        # 執行圖（Graph）
        final_state = self.graph.invoke(initial_state)
        # 取得最終的需求文件
        return final_state["requirements_doc"]


# 執行方式:
# poetry run python -m documentation_agent.main --task "請在此輸入使用者請求"
# 執行範例：
# poetry run python -m documentation_agent.main --task "我想開發一款適用於智慧型手機的健康管理 App"
def main():
    import argparse

    # 建立命令列參數解析器
    parser = argparse.ArgumentParser(
        description="根據使用者需求生成需求定義"
    )
    # 新增 "task" 參數
    parser.add_argument(
        "--task",
        type=str,
        help="請描述你想製作的應用程式",
    )
    # 新增 "k" 參數
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="請設定要生成的人物設定數量(預設:5)",
    )
    # 解析命令列參數
    args = parser.parse_args()

    # 初始化 ChatOpenAI 模型
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    # 初始化需求文件生成 AI 代理
    agent = DocumentationAgent(llm=llm, k=args.k)
    # 執行代理並取得最終輸出
    final_output = agent.run(user_request=args.task)

    # 顯示最終輸出結果
    print(final_output)


if __name__ == "__main__":
    main()

# Interbiews
# {
#     'persona': {
#         'name': 'Alice Chen',
#         'background': 'Alice is a 28-year-old software engineer living in San Francisco. She is tech-savvy and enjoys using apps to track her fitness and health goals. Alice is particularly interested in integrating wearable   
# technology with health apps to provide real-time data.'
#     },
#     'question': 'Alice，作為一名對科技非常熟悉的軟體工程師，您在開發健康管理 App 時，如何看待將可穿戴技術整合進應用程式以提供即時數據的挑戰和機會？您認為哪些功能對於提升用戶的健康追蹤體驗最為重要？',
#     'answer': '整合可穿戴技術到健康管理 App
# 中，確實是一個充滿挑戰和機會的領域。從挑戰的角度來看，首先是數據的準確性和一致性。不同的可穿戴設備可能會使用不同的感測器和算法來收集數據，因此確保這些數據的準確性和一致性是非常重要的。此外，還有隱私和安全問題，因為健康數據是非 
# ，必須確保在傳輸和存儲過程中的安全性。\n\n另一方面，這也帶來了許多機會。即時數據可以讓用戶更好地了解自己的健康狀況，並做出及時的調整。例如，心率、步數、睡眠質量等數據可以幫助用戶制定更有效的健身計劃，甚至可以在異常情況下提供警 
# 示。\n\n至於哪些功能對於提升用戶的健康追蹤體驗最為重要，我認為以下幾點是關鍵：\n\n1. **個性化建議**：根據用戶的歷史數據和目標，提供個性化的健康建議和提醒。\n\n2.
# **即時反饋**：提供即時的數據分析和反饋，讓用戶能夠立即了解自己的健康狀況。\n\n3. **整合性**：能夠與其他健康應用和設備無縫整合，提供一個全面的健康管理平台。\n\n4.
# **社交互動**：增加社交功能，讓用戶可以與朋友分享進展，互相激勵。\n\n5. **目標設定和追蹤**：幫助用戶設定可達成的健康目標，並追蹤其進展。\n\n這些功能不僅能提升用戶的體驗，還能促進他們的健康管理習慣。'
# }
# {
#     'persona': {
#         'name': 'Raj Patel',
#         'background': 'Raj is a 45-year-old high school teacher from Mumbai. He is not very tech-savvy but is eager to learn more about using technology to improve his health. Raj is interested in simple, user-friendly apps    
# that can help him manage his diet and exercise routine.'
#     },
#     'question': '考慮到Raj
# Patel的背景和需求，以下是一個針對他的問題：\n\n"Raj，作為一名不太精通科技的高中教師，您希望這款健康管理App具備哪些簡單易用的功能來幫助您管理飲食和運動計劃？您是否有特定的健康目標或偏好，例如追蹤每日步數、記錄飲食日誌或接收健康 
# 示，這些功能會讓您更容易使用這款App？"',
#     'answer': '作為一名不太精通科技的高中教師，我希望這款健康管理App能夠具備一些簡單易用的功能，讓我能夠輕鬆管理我的飲食和運動計劃。以下是我希望看到的一些功能：\n\n1.
# **步數追蹤**：一個自動計算每日步數的功能，這樣我就不需要手動輸入數據，只需攜帶手機即可。\n\n2. **飲食日誌**：一個簡單的飲食記錄功能，最好能夠通過拍照或選擇常見食物來快速記錄我的餐點。\n\n3.
# **健康提示**：每天提供一些簡單的健康建議或小貼士，幫助我保持健康的生活方式。\n\n4. **運動計劃**：提供一些簡單的運動建議或計劃，適合像我這樣的初學者，並且不需要太多設備。\n\n5.
# **提醒功能**：能夠設置提醒，提醒我喝水、運動或記錄飲食，這樣我就不會忘記。\n\n我的健康目標主要是保持健康的體重和提高日常活動量。我希望這款App能夠幫助我更有規律地管理我的健康，而不需要太多複雜的操作。'
# }
# {
#     'persona': {
#         'name': 'Maria Gonzalez',
#         'background': 'Maria is a 35-year-old stay-at-home mom in Mexico City. She is moderately familiar with smartphone apps and is looking for a health management app that can help her balance her busy schedule with
# maintaining a healthy lifestyle. Maria is particularly interested in meal planning and stress management features.'
#     },
#     'question': 'Maria，作為一位在墨西哥城的全職媽媽，您在尋找一款健康管理 App 時，哪些功能對您來說是最重要的？特別是考慮到您對餐食計劃和壓力管理的興趣，您希望這款 App 如何幫助您在繁忙的日程中更好地維持健康的生活方式？',       
#     'answer': '對我來說，作為一位全職媽媽，健康管理 App 最重要的功能包括：\n\n1. **餐食計劃**：我希望這款 App
# 能提供簡單易用的餐食計劃功能，最好能根據我的家庭喜好和營養需求推薦健康的食譜。能夠生成購物清單，甚至提供一些快速、健康的食譜選擇，對我來說會非常有幫助。\n\n2. **壓力管理**：我希望 App
# 能提供一些壓力管理工具，比如冥想指導、呼吸練習或是一些放鬆音樂。這樣我可以在忙碌的日常中抽出幾分鐘來放鬆自己。\n\n3.
# **時間管理**：一個能幫助我安排每日健康活動的日曆或提醒功能，確保我能夠在繁忙的日程中找到時間來運動或進行其他健康活動。\n\n4.
# **個性化建議**：根據我的健康目標和生活方式，提供個性化的建議和提醒，這樣我可以更有針對性地改善我的健康狀況。\n\n5. **社群支持**：如果 App
# 有一個社群功能，讓我能夠與其他用戶分享經驗和獲得支持，那會非常棒，因為這樣我可以從其他媽媽那裡學到更多的技巧和建議。\n\n這些功能能夠幫助我在繁忙的生活中更好地維持健康的生活方式，讓我能夠更有效地管理時間和壓力，同時確保全家人的 
# 食健康。'
# }
# {
#     'persona': {
#         'name': 'John Smith',
#         'background': 'John is a 60-year-old retired accountant from Sydney. He is relatively new to using smartphones but is keen on using technology to monitor his health, especially to keep track of his medications and      
# doctor appointments. John values apps that offer clear instructions and easy navigation.'
#     },
#     'question': '考慮到 John Smith 的背景和需求，您認為在設計這款健康管理 App 時，哪些功能對於像 John 這樣的使用者來說是最重要的？特別是考慮到他對智慧型手機的使用經驗有限，您會如何確保 App
# 的介面簡單易懂，並且能夠有效地幫助他追蹤藥物和醫生預約？',
#     'answer': '考慮到我的背景和需求，設計一款健康管理 App 時，以下功能對我來說會非常重要：\n\n1.
# **簡單的藥物提醒系統**：這個功能應該能夠讓我輕鬆地輸入和管理我的藥物清單，並設置提醒時間。最好能有一個簡單的步驟來添加藥物，比如掃描藥瓶上的條碼。\n\n2.
# **醫生預約管理**：這個功能應該能夠讓我記錄和查看即將到來的醫生預約，並設置提醒。能夠同步到我的日曆會更好，這樣我就不會錯過任何約會。\n\n3. **用戶友好的介面**：App
# 的介面應該簡單明瞭，使用大圖標和清晰的文字。導航應該直觀，讓我能夠輕鬆找到我需要的功能。\n\n4. **清晰的指導和說明**：每個功能應該有簡單的說明或教程，最好是一步一步的指導，幫助我快速上手。\n\n5.
# **緊急聯絡資訊**：能夠儲存和快速訪問緊急聯絡人的資訊，這樣在需要時可以迅速聯繫到他們。\n\n6.
# **數據備份和同步**：確保我的健康數據能夠安全地備份，並在不同設備間同步，這樣即使更換設備也不會丟失重要資訊。\n\n為了確保這些功能能夠有效地幫助我，App
# 應該進行用戶測試，特別是針對像我這樣的初學者，確保每個步驟都易於理解和操作。提供一個客服支持或幫助中心也是很有幫助的，這樣我在遇到問題時可以得到及時的協助。'
# }
# {
#     'persona': {
#         'name': 'Fatima Al-Mansouri',
#         'background': 'Fatima is a 22-year-old university student in Dubai studying nutrition. She is highly proficient with technology and is interested in apps that offer advanced features like personalized health insights   
# and integration with social media to share her health journey with friends.'
#     },
#     'question': 'Fatima，作為一名在杜拜學習營養學的22歲大學生，您對於一款健康管理App有哪些特定的功能需求？特別是，您希望這款App如何提供個人化的健康見解，以及如何與社交媒體整合以便您能夠與朋友分享您的健康旅程？',
#     'answer':
# '作為一名在杜拜學習營養學的學生，我對健康管理App有一些特定的功能需求。首先，我希望這款App能夠提供個人化的健康見解。這意味著它應該能夠根據我的個人資料、健康目標和日常活動來提供量身定制的建議。例如，根據我的飲食習慣和運動量，App 
# 可以給出營養建議或提醒我補充特定的維生素和礦物質。\n\n此外，這款App應該能夠與可穿戴設備整合，這樣我就能夠實時跟蹤我的健康數據，比如步數、心率和睡眠質量。這些數據可以用來生成更精確的健康見解，幫助我更好地管理我的健康狀況。\n\n  
# 在社交媒體整合方面，我希望這款App能夠讓我輕鬆地分享我的健康旅程。這可以包括分享我的運動成就、健康食譜或者是達成某個健康目標的里程碑。我希望能夠自定義分享的內容，並選擇分享給哪些朋友或群組。此外，App還可以提供一個社區功能，讓用 
# 間可以互相支持和激勵，這樣我就能夠從朋友和其他用戶那裡獲得更多的動力和靈感。\n\n總之，我希望這款App不僅僅是一個工具，而是一個能夠幫助我實現健康目標的夥伴，同時也能讓我與朋友分享這段旅程的點滴。'
# }

# EvaluationResult
# 使用者的需求以及訪談結果，已充分展現出目標使用者對於健康管理 App 在需求、偏好與期待上的全面理解。訪談涵蓋了多元化的人物設定，每一位人物皆具備其特定的需求與挑戰，且皆有清楚的描述。各人物所期望的關鍵功能與特色也被詳細說明，
# 包括個人化設定、資料同步、使用者介面簡潔性、與穿戴式裝置的整合，以及進階的追蹤與分析工具等。這些洞察已足以撰寫一份完整且詳盡的需求文件，能夠滿足不同使用者族群的需求，確保該應用程式同時具備多功能性與良好的使用體驗。


# Final Result
# # 健康管理 App 需求文件

# ## 1. 專案概述
# 本專案旨在開發一款適用於智慧型手機的健康管理 App，旨在幫助用戶更有效地管理健康和健身目標。此 App 將整合多種健康數據來源，提供個性化建議，並支持多種創新技術，以滿足不同用戶的需求。

# ## 2. 主要功能
# ### 2.1 整合性
# - 整合步數、心率、睡眠質量等多種健康數據來源。
# - 支持與智能手錶或健身追蹤器的無縫連接。

# ### 2.2 個性化建議
# - 基於用戶的健康數據和目標，提供個性化的健身和飲食建議。
# - 利用人工智慧分析數據，提供精準的建議和預測。

# ### 2.3 易用性
# - 界面設計簡潔直觀，操作方便。
# - 支持語音助手整合，提供語音指令功能。

# ### 2.4 提醒和通知
# - 設置提醒來幫助用戶記得喝水、運動或休息。
# - 提供藥物提醒和預約管理功能。

# ### 2.5 社交功能
# - 支持與朋友分享進展或參加挑戰。
# - 提供社群互動功能，與其他用戶交流經驗。

# ### 2.6 創新技術
# - 支持虛擬現實（VR）和增強現實（AR）技術。
# - 提供AR跑步伴侶和AI建議功能。

# ## 3. 非功能性需求
# - **性能**：App 應能夠快速響應用戶操作，並在後台高效運行。
# - **安全性**：確保用戶數據的隱私和安全，符合相關法律法規。
# - **兼容性**：支持主流的智慧型手機操作系統（iOS 和 Android）。

# ## 4. 限制條件
# - 需考慮不同用戶的技術熟悉程度，提供多種使用模式。
# - 需確保與多種健康設備的兼容性。

# ## 5. 目標使用者
# - **Alice Chen**：28歲，科技精通的軟體工程師，尋求整合性和個性化建議。
# - **Raj Patel**：45歲，小企業主，尋求簡單易用的飲食和運動追蹤。
# - **Maria Gonzalez**：34歲，護士，尋求壓力管理和睡眠質量分析。
# - **Tommy Nguyen**：19歲，大學生，尋求營養和運動追蹤。
# - **Fatima Al-Mansouri**：60歲，退休教師，尋求藥物和預約管理。

# ## 6. 優先順序
# 1. 整合性和個性化建議功能。
# 2. 易用性和提醒通知功能。
# 3. 創新技術的應用（如AI、AR）。
# 4. 社交功能和社群互動。

# ## 7. 風險與因應對策
# - **技術風險**：可能面臨與多種健康設備的整合困難。應提前進行技術調研，確保兼容性。
# - **市場風險**：用戶需求多樣，可能導致功能過於複雜。應進行用戶調研，確保功能設計符合目標用戶需求。
# - **安全風險**：用戶數據的隱私和安全需高度重視。應採用先進的加密技術，並定期進行安全測試。