// A plan is part of the conversation (MilaAssistant.md 3.1, revised 2026-11-26): the session's plan is kept on the call
// that proposed it, each step as it stands, and nothing of it is pinned above the bar.

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { createInitialState, reduce } from '../state.js';

function apply( messages )
{
    let state = createInitialState();

    for ( const message of messages )
        state = reduce( state, message );

    return state;
}

function sessionState( plan )
{
    return { v: 1, type: 'session.state', model: 'gemma-4-12b-it-q4_0', busy: false, plan, modalities: [],
        streams: [ { stream_id: 's', parent_stream_id: null, conversation_id: 'c1', busy: false } ] };
}

const begin = { type: 'turn.begin', turn_id: 't1', conversation_id: 'c1' };
const proposed = { type: 'turn.tool_call', turn_id: 't1', round: 1, call_id: 'p1', name: 'propose_plan', summary: 'Proposing a plan',
    arguments: JSON.stringify( { steps: '1. Write main.cpp\n2. Build it\n3. Run it' } ) };
const approved = { type: 'turn.tool_result', turn_id: 't1', call_id: 'p1', result: 'The user approved the plan of 3 steps.', is_error: false };

function planBlock( state )
{
    return state.transcript[ 0 ].blocks.find( ( block ) => block.callId === 'p1' );
}

function thePlanIsKeptOnTheCallThatProposedItAsItStands()
{
    const plan = [ { text: 'Write main.cpp', status: 'done' }, { text: 'Build it', status: 'in_progress' }, { text: 'Run it', status: 'pending' } ];
    const state = apply( [ sessionState( null ), begin, proposed, approved, sessionState( plan ) ] );

    assert.deepEqual( planBlock( state ).plan, plan );
}

test( 'the plan is kept on the call that proposed it, as it stands', thePlanIsKeptOnTheCallThatProposedItAsItStands );

function aDeclinedProposalIsNotTheChatsPlan()
{
    const plan = [ { text: 'Write main.cpp', status: 'pending' } ];
    const declined = { type: 'turn.tool_result', turn_id: 't1', call_id: 'p1', result: 'The user declined the plan.', is_error: false };
    let state = apply( [ sessionState( null ), begin, proposed, declined ] );

    // Marked as the window marks a call the person declined.
    state = { ...state, transcript: state.transcript.map( ( entry ) => ( { ...entry,
        blocks: entry.blocks.map( ( block ) => block.callId === 'p1' ? { ...block, decision: 'deny' } : block ) } ) ) };
    state = reduce( state, sessionState( plan ) );

    assert.equal( planBlock( state ).plan, undefined );
}

test( 'a declined proposal is not the chat\'s plan', aDeclinedProposalIsNotTheChatsPlan );

function aLaterPlanMarksTheLaterProposal()
{
    const second = { type: 'turn.tool_call', turn_id: 't2', round: 1, call_id: 'p2', name: 'propose_plan', summary: 'Proposing a plan',
        arguments: JSON.stringify( { steps: 'Add a test' } ) };
    const secondApproved = { type: 'turn.tool_result', turn_id: 't2', call_id: 'p2', result: 'The user approved the plan of 1 steps.', is_error: false };
    const plan = [ { text: 'Add a test', status: 'pending' } ];
    const state = apply( [ sessionState( null ), begin, proposed, approved, { type: 'turn.begin', turn_id: 't2', conversation_id: 'c1' },
        second, secondApproved, sessionState( plan ) ] );
    const later = state.transcript.at( -1 ).blocks.find( ( block ) => block.callId === 'p2' );

    assert.deepEqual( later.plan, plan );
    assert.equal( planBlock( state ).plan, undefined, 'the earlier proposal keeps its steps as proposed' );
}

test( 'a later plan marks the later proposal, not the earlier one', aLaterPlanMarksTheLaterProposal );
