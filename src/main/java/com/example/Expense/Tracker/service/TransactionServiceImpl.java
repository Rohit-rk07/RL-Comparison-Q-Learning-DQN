package com.example.Expense.Tracker.service;

import com.example.Expense.Tracker.model.Transaction;
import com.example.Expense.Tracker.model.User;
import com.example.Expense.Tracker.repository.TransactionRepository;
import com.example.Expense.Tracker.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class TransactionServiceImpl implements TransactionService {

    @Autowired
    private TransactionRepository transactionRepository;

    @Autowired
    private UserRepository userRepository;

    @Override
    public Transaction addTransaction(Transaction transaction, String username) {
        // Fetch user using username
        User user = userRepository.findByUsername(username);
        if (user == null) {
            throw new RuntimeException("User not found with username: " + username);
        }

        // Associate user with transaction
        transaction.setUser(user);
        return transactionRepository.save(transaction);
    }

    @Override
    public List<Transaction> getAllTransactions(String username) {
        return transactionRepository.findByUserUsername(username);
    }

    @Override
    public void deleteTransaction(Long id, String username) {
        Transaction transaction = transactionRepository.findByIdAndUserUsername(id, username);
        if (transaction != null) {
            transactionRepository.delete(transaction);
        } else {
            throw new RuntimeException("Transaction not found or does not belong to the user: " + username);
        }
    }
}
