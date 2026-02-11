package com.example.Expense.Tracker.controller;

import com.example.Expense.Tracker.model.Transaction;
import com.example.Expense.Tracker.service.TransactionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/ExpTrack/transactions")
@CrossOrigin
public class TransactionController {

    @Autowired
    private TransactionService transactionService;

    // Add transaction for given username
    @PostMapping("/{username}")
    public Transaction addTransaction(@RequestBody Transaction transaction, @PathVariable String username) {
        return transactionService.addTransaction(transaction, username);
    }

    // Get all transactions for given username
    @GetMapping("/{username}")
    public List<Transaction> getAllTransactions(@PathVariable String username) {
        return transactionService.getAllTransactions(username);
    }

    // Delete specific transaction by id and username
    @DeleteMapping("/{username}/{id}")
    public void deleteTransaction(@PathVariable String username, @PathVariable Long id) {
        transactionService.deleteTransaction(id, username);
    }
}
