import hashlib
import json
import time
from typing import List, Dict, Any

class Transaction:
    def __init__(self, sender: str, recipient: str, amount: float):
        self.sender = sender
        self.recipient = recipient
        self.amount = amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            'sender': self.sender,
            'recipient': self.recipient,
            'amount': self.amount
        }

class Block:
    def __init__(self, index: int, previous_hash: str, timestamp: float, transactions: List[Transaction]):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.transactions = transactions
        self.nonce = 0

    def compute_hash(self) -> str:
        block_string = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'transactions': [txn.to_dict() for txn in self.transactions],
            'nonce': self.nonce
        }

class Blockchain:
    def __init__(self):
        self.chain: List[Block] = []
        self.current_transactions: List[Transaction] = []
        self.create_block(previous_hash='1', nonce=0)

    def create_block(self, nonce: int, previous_hash: str) -> Block:
        block = Block(index=len(self.chain) + 1,
                      previous_hash=previous_hash,
                      timestamp=time.time(),
                      transactions=self.current_transactions)
        block.nonce = nonce
        self.chain.append(block)
        self.current_transactions = []
        return block

    def add_transaction(self, transaction: Transaction) -> int:
        self.current_transactions.append(transaction)
        return self.last_block.index + 1

    @property
    def last_block(self) -> Block:
        return self.chain[-1]

    def proof_of_work(self, last_proof: int) -> int:
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof: int, proof: int) -> bool:
        guess = f"{last_proof}{proof}".encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    def is_chain_valid(self, chain: List[Block]) -> bool:
        previous_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            if block.previous_hash != previous_block.compute_hash():
                return False
            if not self.valid_proof(previous_block.nonce, block.nonce):
                return False
            previous_block = block
            current_index += 1
        return True

def main():
    blockchain = Blockchain()

    # Create a sample transaction
    txn1 = Transaction("Alice", "Bob", 50)
    blockchain.add_transaction(txn1)
    proof = blockchain.proof_of_work(blockchain.last_block.nonce)
    previous_hash = blockchain.last_block.compute_hash()
    blockchain.create_block(proof, previous_hash)

    txn2 = Transaction("Bob", "Charlie", 25)
    blockchain.add_transaction(txn2)
    proof = blockchain.proof_of_work(blockchain.last_block.nonce)
    previous_hash = blockchain.last_block.compute_hash()
    blockchain.create_block(proof, previous_hash)

    for block in blockchain.chain:
        print(f"Block {block.index}: Hash: {block.compute_hash()}")
        for txn in block.transactions:
            print(f"Transaction: {txn.to_dict()}")

if __name__ == '__main__':
    main()