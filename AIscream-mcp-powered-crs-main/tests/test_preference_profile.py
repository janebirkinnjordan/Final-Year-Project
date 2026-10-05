from backend.app.preference_profile import build_preference_profile, detect_target_domains


def test_cross_domain_profile_extracts_movie_and_shared_preferences():
    history = [
        {'role': 'user', 'content': 'I like emotional sci-fi movies like Interstellar and Arrival.'},
        {'role': 'assistant', 'content': 'Got it.'},
        {'role': 'user', 'content': 'Recommend books.'},
    ]

    profile = build_preference_profile(history)

    assert profile.movie_preferences
    assert 'emotional' in profile.shared_semantic_preferences
    assert 'sci-fi' in profile.shared_semantic_preferences
    assert detect_target_domains('Recommend books.') == ['books']
