import unittest
from unittest.mock import patch
import streamlit as st
from initialize import adjust_seach_kwargs_num_based_on_csv

class TestDynamicSeachKwargsNum(unittest.TestCase):

    def setUp(self):
        """テスト前の初期化処理"""
        st.session_state.clear()

    @patch("streamlit.session_state")
    def test_csv_reference_adjustment(self, mock_session_state):
        """CSV参照時にseach_kwargs_numが動的に調整されることを確認"""
        mock_session_state.get.return_value = "CSVを参照してください"
        result = adjust_seach_kwargs_num_based_on_csv("CSVを参照してください")
        self.assertEqual(result, 10, "CSV参照時にseach_kwargs_numが10に設定されるべき")

    @patch("streamlit.session_state")
    def test_non_csv_reference(self, mock_session_state):
        """CSV参照がない場合にデフォルト値が使用されることを確認"""
        mock_session_state.get.return_value = "通常の質問"
        result = adjust_seach_kwargs_num_based_on_csv("通常の質問")
        self.assertEqual(result, 5, "CSV参照がない場合、デフォルト値5が使用されるべき")

if __name__ == "__main__":
    unittest.main()