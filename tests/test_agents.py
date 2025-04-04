import unittest
from src.agents import create_agents, create_group_chat

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.model = "gpt-4"
        self.work_dir = "test_output"

    def test_create_agents(self):
        agents = create_agents(
            api_key=self.api_key,
            model=self.model,
            work_dir=self.work_dir
        )
        self.assertIn('Code_generator', agents)
        self.assertIn('Code_executor', agents)
        self.assertIn('Critic_agent', agents)
        self.assertIn('Comparer', agents)

    def test_create_group_chat(self):
        agents = create_agents(
            api_key=self.api_key,
            model=self.model,
            work_dir=self.work_dir
        )
        group_chat = create_group_chat(agents)
        self.assertIsNotNone(group_chat)
        self.assertTrue(hasattr(group_chat, 'messages'))