package com.example.Expense.Tracker.repository;

import com.example.Expense.Tracker.model.Transaction;
import com.example.Expense.Tracker.model.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface TransactionRepository extends JpaRepository<Transaction, Long> {
    List<Transaction> findByUserUsername(String username);
    Transaction findByIdAndUserUsername(Long id, String username);
}
