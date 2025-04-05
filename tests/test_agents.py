import unittest
from unittest.mock import patch, MagicMock
import os
from src.agents import create_agents, create_group_chat, setup_llm_config, get_api_key

class TestAgents(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_vars = {
            'OPENAI_API_KEY': 'test-openai-key',
            'AZURE_OPENAI_API_KEY': 'test-azure-key',
            'AZURE_OPENAI_ENDPOINT': 'https://test.openai.azure.com',
            'AZURE_OPENAI_DEPLOYMENT': 'test-deployment',
            'GEMINI_API_KEY': 'test-gemini-key',
            'GROQ_API_KEY': 'test-groq-key'
        }
        self.patcher = patch.dict(os.environ, self.env_vars)
        self.patcher.start()

        self.work_dir = "test_output"

    def tearDown(self):
        self.patcher.stop()

    def test_get_api_key(self):
        self.assertEqual(get_api_key('openai'), 'test-openai-key')
        self.assertEqual(get_api_key('azure'), 'test-azure-key')
        self.assertEqual(get_api_key('gemini'), 'test-gemini-key')
        self.assertEqual(get_api_key('groq'), 'test-groq-key')

        with self.assertRaises(ValueError):
            get_api_key('invalid_provider')

    def test_setup_llm_config_openai(self):
        config = setup_llm_config("openai", self.api_key, "gpt-4")
        self.assertEqual(config[0]["model"], "gpt-4")
        self.assertEqual(config[0]["api_key"], self.api_key)

    def test_setup_llm_config_azure(self):
        config = setup_llm_config(
            "azure",
            self.api_key,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment
        )
        self.assertEqual(config[0]["model"], self.azure_deployment)
        self.assertEqual(config[0]["api_key"], self.api_key)
        self.assertEqual(config[0]["azure_endpoint"], self.azure_endpoint)
        self.assertEqual(config[0]["provider"], "azure")

    def test_setup_llm_config_azure_missing_params(self):
        with self.assertRaises(ValueError):
            setup_llm_config("azure", self.api_key)

    def test_setup_llm_config_gemini(self):
        with patch('google.generativeai.configure') as mock_configure:
            config = setup_llm_config("gemini", self.api_key, "gemini-pro")
            self.assertEqual(config[0]["model"], "gemini-pro")
            self.assertEqual(config[0]["api_key"], self.api_key)
            self.assertEqual(config[0]["provider"], "gemini")
            mock_configure.assert_called_once_with(api_key=self.api_key)

    def test_setup_llm_config_groq(self):
        config = setup_llm_config("groq", self.api_key, "mixtral-8x7b-32768")
        self.assertEqual(config[0]["model"], "mixtral-8x7b-32768")
        self.assertEqual(config[0]["api_key"], self.api_key)
        self.assertEqual(config[0]["provider"], "groq")

    def test_setup_llm_config_invalid_provider(self):
        with self.assertRaises(ValueError):
            setup_llm_config("invalid_provider", self.api_key)

    def test_create_agents_openai(self):
        agents = create_agents(
            api_key=self.api_key,
            provider="openai",
            model="gpt-4",
            work_dir=self.work_dir
        )
        self._verify_agents(agents, "openai")

    def test_create_agents_azure(self):
        agents = create_agents(
            api_key=self.api_key,
            provider="azure",
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            work_dir=self.work_dir
        )
        self._verify_agents(agents, "azure")

    def test_create_agents_gemini(self):
        with patch('google.generativeai.configure'):
            agents = create_agents(
                api_key=self.api_key,
                provider="gemini",
                model="gemini-pro",
                work_dir=self.work_dir
            )
            self._verify_agents(agents, "gemini")

    def test_create_agents_groq(self):
        agents = create_agents(
            api_key=self.api_key,
            provider="groq",
            model="mixtral-8x7b-32768",
            work_dir=self.work_dir
        )
        self._verify_agents(agents, "groq")

    def _verify_agents(self, agents, expected_provider):
        """Helper method to verify agent creation"""
        self.assertIn('code_generator', agents)
        self.assertIn('code_executor', agents)
        self.assertIn('critic', agents)
        self.assertIn('comparer', agents)
        self.assertIn('forecasting_agent', agents)

        # Check provider configuration
        self.assertEqual(
            agents['code_generator'].llm_config["provider"],
            expected_provider
        )

    def test_create_group_chat(self):
        for provider in ["openai", "azure", "gemini", "groq"]:
            with self.subTest(provider=provider):
                kwargs = {
                    "api_key": self.api_key,
                    "provider": provider,
                    "work_dir": self.work_dir
                }

                if provider == "azure":
                    kwargs.update({
                        "azure_endpoint": self.azure_endpoint,
                        "azure_deployment": self.azure_deployment
                    })

                agents = create_agents(**kwargs)
                group_chat = create_group_chat(agents, provider=provider)
                self.assertIsNotNone(group_chat)
                self.assertEqual(
                    group_chat.llm_config["provider"],
                    provider
                )

if __name__ == '__main__':
    unittest.main()


