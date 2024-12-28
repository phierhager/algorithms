from collections import deque, defaultdict
import heapq


# Two pointers: one input, opposite ends
def two_pointers_opposite_ends(arr):
    i, j = 0, len(arr) - 1
    result = []
    while i < j:
        result.append((arr[i], arr[j]))
        i += 1
        j -= 1
    return result


# Two pointers: two inputs, exhaust both
def two_pointers_two_inputs(arr1, arr2):
    i, j = 0, 0
    result = []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result


# Sliding window
def sliding_window(arr, k):
    result = []
    window_sum = sum(arr[:k])
    result.append(window_sum)
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        result.append(window_sum)
    return result


# Build a prefix sum
def prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix


# Efficient string building
def efficient_string_build(parts):
    return "".join(parts)


# Linked list: fast and slow pointer
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False


# Reversing a linked list
def reverse_linked_list(head):
    prev, curr = None, head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev


# Find number of subarrays that fit an exact criteria
def count_subarrays_with_sum(arr, target):
    prefix_sum = {0: 1}
    current_sum = 0
    count = 0
    for num in arr:
        current_sum += num
        count += prefix_sum.get(current_sum - target, 0)
        prefix_sum[current_sum] = prefix_sum.get(current_sum, 0) + 1
    return count


# Monotonic increasing stack
def monotonic_stack(arr):
    stack = []
    result = []
    for value in arr:
        while stack and stack[-1] > value:
            stack.pop()
        stack.append(value)
        result.append(list(stack))
    return result


# Tree node for trees
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Binary tree: DFS (recursive)
def dfs_recursive(node):
    if not node:
        return []
    return [node.val] + dfs_recursive(node.left) + dfs_recursive(node.right)


# Binary tree: DFS (iterative)
def dfs_iterative(root):
    stack, result = [root], []
    while stack:
        node = stack.pop()
        if node:
            result.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return result


# Binary tree: BFS
def bfs(root):
    queue, result = deque([root]), []
    while queue:
        node = queue.popleft()
        if node:
            result.append(node.val)
            queue.append(node.left)
            queue.append(node.right)
    return result


# Graph: DFS (recursive)
def graph_dfs_recursive(graph, node, visited):
    if node in visited:
        return []
    visited.add(node)
    result = [node]
    for neighbor in graph[node]:
        result.extend(graph_dfs_recursive(graph, neighbor, visited))
    return result


# Graph: DFS (iterative)
def graph_dfs_iterative(graph, start):
    stack, visited, result = [start], set(), []
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)
            stack.extend(reversed(graph[node]))
    return result


# Graph: BFS
def graph_bfs(graph, start):
    queue, visited, result = deque([start]), set(), []
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            result.append(node)
            queue.extend(graph[node])
    return result


# Find top k elements with heap
def top_k_elements(arr, k):
    return heapq.nlargest(k, arr)


# Binary search
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# Binary search: duplicate elements, left-most insertion point
def leftmost_binary_search(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    return low


# Binary search: duplicate elements, right-most insertion point
def rightmost_binary_search(arr, target):
    low, high = 0, len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return low - 1


# Backtracking
def backtracking(nums, path, result):
    if not nums:
        result.append(path)
    for i in range(len(nums)):
        backtracking(nums[:i] + nums[i + 1 :], path + [nums[i]], result)


# Dynamic programming: top-down memoization
def fib_memo(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


# Build a trie
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word


# Dijkstra's algorithm
def dijkstra(graph, start):
    pq = [(0, start)]  # (distance, node)
    distances = {node: float("inf") for node in graph}
    distances[start] = 0

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
