from reporecall.retrieval import EngineeringTokenizer


def test_ordinary_prose_produces_case_insensitive_lexical_terms():
    tokenizer = EngineeringTokenizer()

    tokens = tokenizer.tokenize(
        "Database connection failed after worker restart. DATABASE"
    )

    assert tokens == (
        "database",
        "connection",
        "failed",
        "after",
        "worker",
        "restart",
        "database",
    )


def test_snake_case_preserves_complete_identifiers_and_components():
    tokens = EngineeringTokenizer().tokenize(
        "retry_worker database_connection_pool"
    )

    assert "retry_worker" in tokens
    assert "retry" in tokens
    assert "worker" in tokens
    assert "database_connection_pool" in tokens
    assert "database" in tokens
    assert "connection" in tokens
    assert "pool" in tokens


def test_camel_and_pascal_case_preserve_complete_identifiers_and_components():
    tokens = EngineeringTokenizer().tokenize(
        "ConnectionPoolManager retryWorkerTask"
    )

    assert "connectionpoolmanager" in tokens
    assert "connection" in tokens
    assert "pool" in tokens
    assert "manager" in tokens
    assert "retryworkertask" in tokens
    assert "retry" in tokens
    assert "worker" in tokens
    assert "task" in tokens


def test_paths_preserve_complete_identity_and_useful_components():
    tokens = EngineeringTokenizer().tokenize(
        "src/database/session.py tests/test_retry_worker.py"
    )

    assert "src/database/session.py" in tokens
    assert "src" in tokens
    assert "database" in tokens
    assert "session.py" in tokens
    assert "session" in tokens
    assert "py" in tokens
    assert "tests/test_retry_worker.py" in tokens
    assert "test_retry_worker.py" in tokens
    assert "test" in tokens
    assert "retry" in tokens
    assert "worker" in tokens


def test_function_syntax_preserves_function_and_qualified_identifiers():
    tokens = EngineeringTokenizer().tokenize(
        "retry_worker() DatabasePool.close()"
    )

    assert "retry_worker" in tokens
    assert "databasepool.close" in tokens
    assert "databasepool" in tokens
    assert "database" in tokens
    assert "pool" in tokens
    assert "close" in tokens


def test_git_sha_survives_as_an_exact_token():
    sha = "9a0b27473cfb40769d1b06ac827241fb89025def"

    assert sha in EngineeringTokenizer().tokenize(sha)


def test_error_codes_preserve_complete_lexical_identity():
    tokens = EngineeringTokenizer().tokenize(
        "HTTP_429 ERR_CONNECTION_RESET ORA-12541 EADDRINUSE"
    )

    assert "http_429" in tokens
    assert "err_connection_reset" in tokens
    assert "ora-12541" in tokens
    assert "eaddrinuse" in tokens


def test_patch_content_preserves_meaningful_identifiers():
    patch = """@@ -10,4 +10,6 @@
- connection.release()
+ connection.close()
"""

    tokens = EngineeringTokenizer().tokenize(patch)

    assert "connection.release" in tokens
    assert "connection.close" in tokens
    assert "connection" in tokens
    assert "release" in tokens
    assert "close" in tokens


def test_stack_trace_preserves_exact_technical_terms():
    stack_trace = """Traceback (most recent call last):
  File "src/database/session.py", line 82, in worker_retry
    DatabasePool.close()
sqlalchemy.exc.TimeoutError: QueuePool limit reached
"""

    tokens = EngineeringTokenizer().tokenize(stack_trace)

    assert "src/database/session.py" in tokens
    assert "worker_retry" in tokens
    assert "databasepool.close" in tokens
    assert "sqlalchemy.exc.timeouterror" in tokens
    assert "sqlalchemy" in tokens
    assert "exc" in tokens
    assert "timeouterror" in tokens


def test_urls_are_reduced_to_hostname_and_path_fragments():
    tokens = EngineeringTokenizer().tokenize(
        "See https://github.com/owner/repo/issues/123?notification=1#event"
    )

    assert "github.com" in tokens
    assert "owner" in tokens
    assert "repo" in tokens
    assert "issues" in tokens
    assert "123" in tokens
    assert "notification" not in tokens
    assert "event" not in tokens


def test_tokenization_is_deterministic_and_does_not_mutate_source_text():
    tokenizer = EngineeringTokenizer()
    source = "ConnectionResetError in retry_worker()"

    first = tokenizer.tokenize(source)
    second = tokenizer.tokenize(source)

    assert first == second
    assert source == "ConnectionResetError in retry_worker()"


def test_punctuation_only_text_produces_no_tokens():
    assert EngineeringTokenizer().tokenize("... () [] {} !!!") == ()
