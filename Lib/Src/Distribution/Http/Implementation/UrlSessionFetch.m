/**
 * @file UrlSessionFetch.m
 * @brief One request through Apple's URL loading: Keychain trust, the Mac's proxy settings, no redirect followed.
 *
 * Objective-C because URL loading is, compiled with ARC, and C at its edge so no C++ crosses into it
 * (HttpClient.md section 9).
 */

#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "UrlSessionFetch.h"

/* Seven days: the loader's own longest, standing for none. */
static const NSTimeInterval kNoTimeout = 7.0 * 24.0 * 60.0 * 60.0;

@interface MilaUrlSessionDelegate : NSObject <NSURLSessionDataDelegate>

- (instancetype)initWithContext:(void*)context head:(MilaUrlSessionHeadCallback)head data:(MilaUrlSessionDataCallback)data;

@property (nonatomic, readonly) dispatch_semaphore_t done;
@property (nonatomic, readonly) BOOL stopped;
@property (nonatomic, readonly, strong) NSError* error;

@end

@implementation MilaUrlSessionDelegate
{
    void* _context;
    MilaUrlSessionHeadCallback _head;
    MilaUrlSessionDataCallback _data;
    BOOL _headReported;
}

- (instancetype)initWithContext:(void*)context head:(MilaUrlSessionHeadCallback)head data:(MilaUrlSessionDataCallback)data
{
    self = [super init];

    if ( self != nil )
    {
        _context = context;
        _head = head;
        _data = data;
        _done = dispatch_semaphore_create( 0 );
    }

    return self;
}

/* Reports the reply once, whichever of the delegate's calls sees it first: a refused redirect is the reply. */
- (void)reportHead:(NSURLResponse*)response
{
    if ( _headReported || ![response isKindOfClass:[NSHTTPURLResponse class]] )
    {
        return;
    }

    _headReported = YES;

    NSHTTPURLResponse* reply = (NSHTTPURLResponse*)response;
    NSString* location = [reply valueForHTTPHeaderField:@"Location"];
    NSString* length = [reply valueForHTTPHeaderField:@"Content-Length"];

    _head( _context, (long)reply.statusCode, location != nil ? location.UTF8String : "",
        length != nil ? strtoull( length.UTF8String, NULL, 10 ) : 0 );
}

- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task
    willPerformHTTPRedirection:(NSHTTPURLResponse*)response newRequest:(NSURLRequest*)request
    completionHandler:(void ( ^ )( NSURLRequest* ))completionHandler
{
    [self reportHead:response];
    completionHandler( nil );
}

- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task
    didReceiveResponse:(NSURLResponse*)response
    completionHandler:(void ( ^ )( NSURLSessionResponseDisposition ))completionHandler
{
    [self reportHead:response];
    completionHandler( NSURLSessionResponseAllow );
}

- (void)URLSession:(NSURLSession*)session dataTask:(NSURLSessionDataTask*)task didReceiveData:(NSData*)data
{
    [self reportHead:task.response];

    if ( _stopped )
    {
        return;
    }

    __block BOOL keep = YES;

    [data enumerateByteRangesUsingBlock:^( const void* bytes, NSRange range, BOOL* stop ) {
        if ( !self->_data( self->_context, (const char*)bytes, range.length ) )
        {
            keep = NO;
            *stop = YES;
        }
    }];

    if ( !keep )
    {
        _stopped = YES;
        [task cancel];
    }
}

- (void)URLSession:(NSURLSession*)session task:(NSURLSessionTask*)task didCompleteWithError:(NSError*)error
{
    [self reportHead:task.response];
    _error = error;
    dispatch_semaphore_signal( _done );
}

@end

static void failWith( MilaUrlSessionOutcome* outcome, const char* message )
{
    outcome->transport_failed = 1;
    snprintf( outcome->message, sizeof( outcome->message ), "%s", message );
}

void milaUrlSessionFetch( const MilaUrlSessionRequest* request, void* context,
    MilaUrlSessionHeadCallback on_head, MilaUrlSessionDataCallback on_data, MilaUrlSessionOutcome* outcome )
{
    memset( outcome, 0, sizeof( *outcome ) );

    @autoreleasepool
    {
        NSString* text = [NSString stringWithUTF8String:request->url];
        NSURL* url = text != nil ? [NSURL URLWithString:text] : nil;

        if ( url == nil || url.scheme == nil || url.host == nil )
        {
            failWith( outcome, "the URL is not one URL loading can read" );

            return;
        }

        NSMutableURLRequest* loading = [NSMutableURLRequest requestWithURL:url];
        BOOL agent = NO;
        BOOL encoding = NO;

        for ( size_t index = 0; index < request->header_count; ++index )
        {
            NSString* name = [NSString stringWithUTF8String:request->headers[ index ].name];
            NSString* value = [NSString stringWithUTF8String:request->headers[ index ].value];

            if ( name == nil || value == nil )
            {
                failWith( outcome, "a header is not UTF-8 text" );

                return;
            }

            agent = agent || [name caseInsensitiveCompare:@"User-Agent"] == NSOrderedSame;
            encoding = encoding || [name caseInsensitiveCompare:@"Accept-Encoding"] == NSOrderedSame;
            [loading addValue:value forHTTPHeaderField:name];
        }

        if ( !agent )
        {
            [loading setValue:@"Mila/0.20" forHTTPHeaderField:@"User-Agent"];
        }

        // Unasked, the loader requests compression and decodes it, so a resumed or hashed download would not see the
        // resource's own bytes; curl sends no Accept-Encoding, which means the same.
        if ( !encoding )
        {
            [loading setValue:@"identity" forHTTPHeaderField:@"Accept-Encoding"];
        }

        if ( request->has_body )
        {
            loading.HTTPMethod = @"POST";
            loading.HTTPBody = [NSData dataWithBytes:request->body length:request->body_length];
        }

        NSURLSessionConfiguration* configuration = [NSURLSessionConfiguration ephemeralSessionConfiguration];
        configuration.URLCache = nil;
        configuration.HTTPCookieStorage = nil;
        configuration.HTTPShouldSetCookies = NO;
        configuration.HTTPCookieAcceptPolicy = NSHTTPCookieAcceptPolicyNever;
        configuration.URLCredentialStorage = nil;
        configuration.requestCachePolicy = NSURLRequestReloadIgnoringLocalCacheData;
        configuration.timeoutIntervalForRequest = request->idle_timeout_seconds > 0 ? (NSTimeInterval)request->idle_timeout_seconds : kNoTimeout;
        configuration.timeoutIntervalForResource = kNoTimeout;

        MilaUrlSessionDelegate* delegate = [[MilaUrlSessionDelegate alloc] initWithContext:context head:on_head data:on_data];
        NSOperationQueue* queue = [[NSOperationQueue alloc] init];
        queue.maxConcurrentOperationCount = 1;

        NSURLSession* session = [NSURLSession sessionWithConfiguration:configuration delegate:delegate delegateQueue:queue];
        NSURLSessionDataTask* task = [session dataTaskWithRequest:loading];
        [task resume];
        dispatch_semaphore_wait( delegate.done, DISPATCH_TIME_FOREVER );
        [session finishTasksAndInvalidate];

        outcome->stopped = delegate.stopped ? 1 : 0;

        if ( delegate.error != nil && !delegate.stopped )
        {
            failWith( outcome, delegate.error.localizedDescription.UTF8String );
        }
    }
}
