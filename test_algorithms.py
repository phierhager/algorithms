import unittest
from algorithms import *


class TestFunctions(unittest.TestCase):
    def test_two_pointers_opposite_ends(self):
        self.assertEqual(
            two_pointers_opposite_ends([1, 2, 3, 4]), [(1, 4), (2, 3)]
        )
        self.assertEqual(two_pointers_opposite_ends([1]), [])

    def test_two_pointers_two_inputs(self):
        self.assertEqual(
            two_pointers_two_inputs([1, 3, 5], [2, 4, 6]), [1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(two_pointers_two_inputs([], [1, 2]), [1, 2])

    def test_sliding_window(self):
        self.assertEqual(sliding_window([1, 2, 3, 4], 2), [3, 5, 7])
        self.assertEqual(sliding_window([1], 1), [1])

    def test_prefix_sum(self):
        self.assertEqual(prefix_sum([1, 2, 3]), [0, 1, 3, 6])
        self.assertEqual(prefix_sum([]), [0])

    def test_efficient_string_build(self):
        self.assertEqual(efficient_string_build(["a", "b", "c"]), "abc")
        self.assertEqual(efficient_string_build([]), "")

    def test_has_cycle(self):
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = head
        self.assertTrue(has_cycle(head))

        head = ListNode(1)
        head.next = ListNode(2)
        self.assertFalse(has_cycle(head))

    def test_reverse_linked_list(self):
        head = ListNode(1, ListNode(2, ListNode(3)))
        reversed_head = reverse_linked_list(head)
        self.assertEqual(reversed_head.val, 3)
        self.assertEqual(reversed_head.next.val, 2)
        self.assertEqual(reversed_head.next.next.val, 1)

    def test_count_subarrays_with_sum(self):
        self.assertEqual(count_subarrays_with_sum([1, 1, 1], 2), 2)
        self.assertEqual(count_subarrays_with_sum([1, 2, 3], 3), 2)

    def test_monotonic_stack(self):
        self.assertEqual(monotonic_stack([3, 2, 1]), [[3], [2], [1]])
        self.assertEqual(monotonic_stack([1, 2, 3]), [[1], [1, 2], [1, 2, 3]])

    def test_dfs_recursive(self):
        root = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual(dfs_recursive(root), [1, 2, 3])

    def test_dfs_iterative(self):
        root = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual(dfs_iterative(root), [1, 2, 3])

    def test_bfs(self):
        root = TreeNode(1, TreeNode(2), TreeNode(3))
        self.assertEqual(bfs(root), [1, 2, 3])

    def test_graph_dfs_recursive(self):
        graph = {1: [2, 3], 2: [], 3: []}
        self.assertEqual(graph_dfs_recursive(graph, 1, set()), [1, 2, 3])

    def test_graph_dfs_iterative(self):
        graph = {1: [2, 3], 2: [], 3: []}
        self.assertEqual(graph_dfs_iterative(graph, 1), [1, 2, 3])

    def test_graph_bfs(self):
        graph = {1: [2, 3], 2: [], 3: []}
        self.assertEqual(graph_bfs(graph, 1), [1, 2, 3])

    def test_top_k_elements(self):
        self.assertEqual(top_k_elements([3, 2, 1, 5, 4], 2), [5, 4])

    def test_binary_search(self):
        self.assertEqual(binary_search([1, 2, 3, 4], 3), 2)
        self.assertEqual(binary_search([1, 2, 3, 4], 5), -1)

    def test_leftmost_binary_search(self):
        self.assertEqual(leftmost_binary_search([1, 2, 2, 3], 2), 1)
        self.assertEqual(leftmost_binary_search([1, 2, 3], 4), 3)

    def test_rightmost_binary_search(self):
        self.assertEqual(rightmost_binary_search([1, 2, 2, 3], 2), 2)
        self.assertEqual(rightmost_binary_search([1, 2, 3], 4), 2)

    def test_backtracking(self):
        result = []
        backtracking([1, 2], [], result)
        self.assertEqual(sorted(result), [[1, 2], [2, 1]])

    def test_fib_memo(self):
        self.assertEqual(fib_memo(10), 55)
        self.assertEqual(fib_memo(0), 0)

    def test_trie(self):
        trie = Trie()
        trie.insert("hello")
        self.assertTrue(trie.search("hello"))
        self.assertFalse(trie.search("hell"))

    def test_dijkstra(self):
        graph = {0: [(1, 1), (2, 4)], 1: [(2, 2), (3, 6)], 2: [(3, 3)], 3: []}
        self.assertEqual(dijkstra(graph, 0), {0: 0, 1: 1, 2: 3, 3: 6})


if __name__ == "__main__":
    unittest.main()
