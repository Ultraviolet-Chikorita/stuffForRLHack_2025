import unittest

from rl_agent import QLearningAgent, WebsiteNavEnv


class _Driver:
    def get(self, url):
        self.last_url = url


class WebsiteNavEnvTests(unittest.TestCase):
    def make_env(self, max_steps=3):
        return WebsiteNavEnv(
            adj={"start": ["mid"], "mid": ["goal", "start"], "goal": []},
            url_to_file={},
            mirror_root=".",
            driver=_Driver(),
            max_steps=max_steps,
            step_penalty=-0.01,
            goal_reward=1.0,
        )

    def test_reaching_goal_ends_episode_with_positive_reward(self):
        env = self.make_env()
        env.reset("start", "goal")
        first = env.step(0)
        second = env.step(0)

        self.assertFalse(first.done)
        self.assertTrue(second.done)
        self.assertEqual(second.state, "goal")
        self.assertAlmostEqual(second.reward, 0.99)

    def test_revisiting_state_is_penalized(self):
        env = self.make_env(max_steps=5)
        env.reset("start", "goal")
        env.step(0)
        revisit = env.step(1)

        self.assertEqual(revisit.state, "start")
        self.assertAlmostEqual(revisit.reward, -0.06)
        self.assertFalse(revisit.done)

    def test_invalid_action_is_rejected(self):
        env = self.make_env()
        env.reset("start", "goal")
        with self.assertRaises(ValueError):
            env.step(2)


class QLearningTests(unittest.TestCase):
    def test_terminal_update_uses_reward_as_target(self):
        env = WebsiteNavEnv(
            adj={"start": ["goal"], "goal": []},
            url_to_file={},
            mirror_root=".",
            driver=_Driver(),
        )
        agent = QLearningAgent(env, learning_rate=0.5)

        agent.update("start", 0, reward=1.0, next_state="goal", done=True)

        self.assertAlmostEqual(agent.q[("start", 0)], 0.5)
        self.assertEqual(agent.total_steps, 1)

    def test_epsilon_decays_to_configured_floor(self):
        env = WebsiteNavEnv(
            adj={"start": []},
            url_to_file={},
            mirror_root=".",
            driver=_Driver(),
        )
        agent = QLearningAgent(
            env,
            epsilon_start=0.2,
            epsilon_end=0.05,
            epsilon_decay_steps=10,
        )
        agent.total_steps = 10_000
        self.assertAlmostEqual(agent._epsilon(), 0.05)


if __name__ == "__main__":
    unittest.main()
