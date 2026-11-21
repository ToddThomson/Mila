// Scenario 6: an image attached to Gemma, which takes images, and described in the reply. Loaded
// with a text-only model, the same attachment is refused before it is sent.

import { singleRoundReply } from './stream.js';

const sample = { name: 'harbour-at-dusk.png', type: 'image/png', sizeBytes: 16276 };

const prompt = 'What\'s in this picture?';

const imageThought = 'An image is attached. I can see open water, a sun half below the horizon, a small sailboat on the left '
    + 'and a red-and-white striped lighthouse on a dark point to the right, with its light on. '
    + 'I will describe the scene first, then the colours and the time of day. The flat shapes look drawn rather than photographed.';

const imageAnswer = 'It\'s a harbour at sunset. A small sailboat with two cream sails sits on calm water to the left, '
    + 'and a red-and-white striped lighthouse stands on a dark, rocky point to the right, its beam reaching out of the frame. '
    + 'Between them the sun is half below the horizon, laying a path of light across the water.\n\n'
    + 'The sky runs from deep blue at the top through orange to pale gold at the horizon, so this is the last few minutes before dusk. '
    + 'The flat colours and clean edges suggest an illustration rather than a photograph.';

const textThought = 'No picture came with the message, so there is nothing to describe yet.';

const textAnswer = 'I don\'t see a picture in your message. Attach one with the paperclip, or drop it onto the window, and ask again.';

function createReply( request, context )
{
    if ( context.attachments.length > 0 )
        return singleRoundReply( imageThought, imageAnswer, 318 );

    return singleRoundReply( textThought, textAnswer, 96 );
}

export const imageAttachment = {
    number: 6,
    title: 'Image attached, then refused',
    model: 'gemma-4-12b-it-fp4',
    prompt,
    sample,
    createReply
};
