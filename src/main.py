import logging
import operator
from typing import Annotated, Any

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

class JavaMigrationState(BaseModel):
    """
    Agentのステータス定義
    """
    original_prompt: str = Field(..., description="入力された目標")
    
    optimized_goal: str = Field(default="", description="最適化された目標")
    optimized_response: str = Field(
        default="", description="最適化されたレスポンス定義"
    )
    tasks: list[str] = Field(default_factory=list, description="実行するタスクのリスト")
    current_task_index: int = Field(default=0, description="現在実行中のタスクの番号")
    results: Annotated[list[str], operator.add] = Field(
        default_factory=list, description="実行済みタスクの結果リスト"
    )
    final_output: str = Field(default="", description="最終的な出力結果")
    error: str = Field(default="", description="exception message")


class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

    @property
    def text(self) -> str:
        return f"{self.description}"


class PassiveGoalCreator:
    """
    目標の詳細化・最適化
    """

    def __init__(
        self,
        llm,
    ):
        self.llm = llm

    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
            "ユーザーの入力を分析し、明確で実行可能な目標を生成してください。\n"
            "要件:\n"
            "1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "   - ユーザーのためのレポートを生成する。\n"
            "3. 決して2.以外の行動を取ってはいけません。\n"
            "ユーザーの入力: {query}"
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=10,
        description="3~5個に分解されたタスク",
    )


class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str):
        prompt = ChatPromptTemplate.from_template(
            "タスク: 与えられた目標を具体的で実行可能なタスクに分解してください。\n"
            "要件:\n"
            "1. 以下の行動だけで目標を達成すること。決して指定された以外の行動をとらないこと。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "2. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。\n"
            "3. タスクは実行可能な順序でリスト化すること。\n"
            "4. タスクは日本語で出力すること。\n"
            "目標: {query}"
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})


class ResponseOptimizer:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。与えられた目標に対して、エージェントが目標にあったレスポンスを返すためのレスポンス仕様を策定してください。",
                ),
                (
                    "human",
                    "以下の手順に従って、レスポンス最適化プロンプトを作成してください：\n\n"
                    "1. 目標分析:\n"
                    "提示された目標を分析し、主要な要素や意図を特定してください。\n\n"
                    "2. レスポンス仕様の策定:\n"
                    "目標達成のための最適なレスポンス仕様を考案してください。トーン、構造、内容の焦点などを考慮に入れてください。\n\n"
                    "3. 具体的な指示の作成:\n"
                    "事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、AIエージェントに対する明確で実行可能な指示を作成してください。あなたの指示によってAIエージェントが実行可能なのは、既に調査済みの結果をまとめることだけです。インターネットへのアクセスはできません。\n\n"
                    "4. 例の提供:\n"
                    "可能であれば、目標に沿ったレスポンスの例を1つ以上含めてください。\n\n"
                    "5. 評価基準の設定:\n"
                    "レスポンスの効果を測定するための基準を定義してください。\n\n"
                    "以下の構造でレスポンス最適化プロンプトを出力してください:\n\n"
                    "目標分析:\n"
                    "[ここに目標の分析結果を記入]\n\n"
                    "レスポンス仕様:\n"
                    "[ここに策定されたレスポンス仕様を記入]\n\n"
                    "AIエージェントへの指示:\n"
                    "[ここにAIエージェントへの具体的な指示を記入]\n\n"
                    "レスポンス例:\n"
                    "[ここにレスポンス例を記入]\n\n"
                    "評価基準:\n"
                    "[ここに評価基準を記入]\n\n"
                    "では、以下の目標に対するレスポンス最適化プロンプトを作成してください:\n"
                    "{query}",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})


class ResultAggregator:
    def __init__(self, llm):
        self.llm = llm

    def run(self, query: str, response_definition: str, results: list[str]) -> str:
        prompt = ChatPromptTemplate.from_template(
            "与えられた目標:\n{query}\n\n"
            "調査結果:\n{results}\n\n"
            "与えられた目標に対し、調査結果を用いて、以下の指示に基づいてレスポンスを生成してください。\n"
            "{response_definition}"
        )
        results_str = "\n\n".join(
            f"Info {i+1}:\n{result}" for i, result in enumerate(results)
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "query": query,
                "results": results_str,
                "response_definition": response_definition,
            }
        )


class TaskExecutor:
    def __init__(self, llm):
        self.llm = llm
        self.tools = [TavilySearchResults(max_results=3)]

    def run(self, task: str) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
                            "次のタスクを実行し、詳細な回答を提供してください。\n\n"
                            f"タスク: {task}\n\n"
                            "要件:\n"
                            "1. 必要に応じて提供されたツールを使用してください。\n"
                            "2. 実行は徹底的かつ包括的に行ってください。\n"
                            "3. 可能な限り具体的な事実やデータを提供してください。\n"
                            "4. 発見した内容を明確に要約してください。\n"
                        ),
                    )
                ]
            }
        )
        return result["messages"][-1].content


class Java17ConverterAgent:
    def __init__(self, llm):
        self.llm = llm
        self.graph = self.build_graph()

    def build_graph(self):
        """
        # --- LangGraph ワークフローの定義 ---
        """
        workflow = StateGraph(JavaMigrationState)
        workflow.add_node("goal_setting", RunnableLambda(self._goal_setting))
        workflow.add_node("decompose_query", RunnableLambda(self._decompose_query))
        workflow.add_node("execute_task", RunnableLambda(self._execute_task))
        workflow.add_node("aggregate_results", RunnableLambda(self._aggregate_results))

        workflow.add_edge(START, "goal_setting")
        workflow.add_edge("goal_setting", "decompose_query")
        workflow.add_edge("decompose_query", "execute_task")
        workflow.add_conditional_edges(
            "execute_task",
            lambda state: state.current_task_index < len(state.tasks),
            {True: "execute_task", False: "aggregate_results"},
        )
        workflow.add_edge("aggregate_results", END)

        graph = workflow.compile()
        return graph

    def run(self, prompt: str):
        result = self.graph.invoke({"original_prompt": prompt})
        return result

    def _goal_setting(self, state: JavaMigrationState):
        """
        入力を最適化された目標に変換
        """
        goal_creator = PassiveGoalCreator(self.llm)
        goal: Goal = goal_creator.run(state.original_prompt)
        optimized_goal = goal.text
        logger.info(f"optimized_goal => {optimized_goal}")

        response_optimizer = ResponseOptimizer(self.llm)
        optimized_response = response_optimizer.run(optimized_goal)
        logger.info(f"optimized_response => {optimized_response}")

        return {
            "optimized_goal": optimized_goal,
            "optimized_response": optimized_response,
        }

    def _decompose_query(self, state: JavaMigrationState) -> dict[str, Any]:
        """
        最適化された目標を元にタスクを細分化
        """
        query_decomposer = QueryDecomposer(self.llm)
        decomposed_tasks: DecomposedTasks = query_decomposer.run(
            query=state.optimized_goal
        )
        logger.info(f"decomposed_tasks => {decomposed_tasks.values}")

        return {"tasks": decomposed_tasks.values}

    def _execute_task(self, state: JavaMigrationState) -> dict[str, Any]:
        """
        各タスクを実行
        """
        current_task = state.tasks[state.current_task_index]
        logger.info(f"current_task =>{current_task}")

        task_executor = TaskExecutor(self.llm)
        result = task_executor.run(task=current_task)
        return {
            "results": [result],
            "current_task_index": state.current_task_index + 1,
        }

    def _aggregate_results(self, state: JavaMigrationState) -> dict[str, Any]:
        """
        最終出力の編集
        """
        result_aggregator = ResultAggregator(self.llm)
        final_output = result_aggregator.run(
            query=state.optimized_goal,
            response_definition=state.optimized_response,
            results=state.results,
        )
        return {"final_output": final_output}


# --- 実行例 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    java_code = """
package jp.example;

import java.util.Date;

public class Hello {

	public static void main(String[] args) {
        Date d = new Date();
        /*
         * 現在日時を出力
         */
        System.out.println(d);
        int num = Integer("16");
        System.out.println(num);
    }
}
"""

    prompt = f"""
次の Java プログラムをJava 17プログラムに変換して下さい。

プログラム
------------------
{java_code}
------------------
"""
    llm = ChatOpenAI(model="gpt-4")

    agent = Java17ConverterAgent(llm=llm)
    # print(agent.graph.get_graph().draw_ascii())
    result_state = agent.run(prompt=prompt)
    print(result_state)
    print("".join(result_state["final_output"]))
