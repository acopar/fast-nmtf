#!/usr/bin/env python

import threading
from queue import Queue
import time
import sys
import traceback
import socket
import os

class Model(object):
    def __init__(self):
        self._lock = threading.RLock()
        self.q = Queue()
        self.db = {}
    
    def get(self, key):
        if key in self.db:
            return self.db[key]
    
    def set(self, key, value):
        with self._lock:
            self.db[key] = value

    def flush_queue(self):
        q = self.q
        while not q.empty():
            el = q.get()
            if el != 0:
                print("Element left in queue", el)

class Task(object):
    def __init__(self, call_func, model):
        self.call_func = call_func
        self.model = model
    
    def __call__(self):
        return self.call_func(self.model)

class WorkerThread(threading.Thread):
    def __init__(self, q, tid, parent_notify):
        super(WorkerThread, self).__init__()
        self.setDaemon(True)
        self._stopper = threading.Event()
        self.running = True
        self.tid = tid
        self.parent_notify = parent_notify
        self.q = q
 
    def stop(self):
        self._stopper.set()
    
    def stopped(self):
        return self._stopper.isSet()
 
    def run(self):
        q = self.q
        while self.stopped() == False:
            try:
                task = q.get(block=True, timeout=1)
                if task == 0:
                    self._stopper.set()
                    self.running = False
                    q.put(task)
                    print("Worker %d exiting" % self.tid)
                    break
                t0 = time.time()
                task()
                q.task_done()
            except Queue.Empty:
                pass
            except Exception as e:
                print(traceback.format_exc())
        
        self.parent_notify(self.tid)
        
class MainThread(threading.Thread):
    def __init__(self, model, n_workers=2):
        super(MainThread, self).__init__()
        self._stopper = threading.Event()
        self._child = threading.Event()
        self.threads = {}
        self.thread_counter = 0
        self.finished_threads = []
        self.n_workers = n_workers
        self.model = model
    
    def run(self):
        self.init_workers(self.n_workers)
        
        while self.stopped() == False:
            if self.zombies():
                self.child_cleanup()
            
            time.sleep(1)
        
        self.cleanup()
    
    def stop(self):
        self._stopper.set()
    
    def init_workers(self, n):
        for i in range(n):
            tid = self.new_tid()
            w = WorkerThread(self.model.q, tid, self.child)
            w.start()
            self.threads[tid] = w
        
    def new_tid(self):
        self.thread_counter += 1
        return self.thread_counter
    
    def child(self, i):
        self.finished_threads.append(i)
        self._child.set()
    
    def child_cleanup(self):
        for i in self.finished_threads:
            self.threads[i].join()
            del self.threads[i]
        
        self.finished_threads = []
        if len(self.threads) == 0:
            self.stop()
        
        self._child.clear()
    
    def cleanup(self):
        for tid, t in self.threads.items():
            t.stop()
        
        for tid, t in self.threads.items():
            t.join(2)
            
        self.model.flush_queue()
    
    def stopped(self):
        return self._stopper.isSet()
    
    def zombies(self):
        return self._child.isSet()
        
