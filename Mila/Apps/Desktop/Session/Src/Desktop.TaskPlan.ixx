/**
 * @file Desktop.TaskPlan.ixx
 * @brief Get Things Done's plan: the steps the model proposed and the person approved, each pending, in progress,
 *        done or skipped (MilaApp.md 3.1).
 */

export module Desktop.TaskPlan;

import std;
import Chat.Json;

namespace Mila::DesktopApp
{
    /**
     * @brief The plan of one conversation, kept by the two tools that change it.
     *
     * `propose_plan` replaces the plan, and the session asks the person first, so no plan is worked through that the
     * person did not see. `update_step` marks one step, and runs without asking, since it changes nothing outside Mila.
     * A failure throws, and the model reads its message.
     */
    export class TaskPlan
    {
    public:

        static constexpr std::string_view kProposeName = "propose_plan";
        static constexpr std::string_view kUpdateName = "update_step";

        /// `steps`: one step a line. The plan becomes those steps, each pending.
        std::string propose( const std::string& arguments_text )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::string text = arguments.at( "steps" ).get<std::string>();
            std::vector<Step> steps;

            for ( const auto line : std::views::split( text, '\n' ) )
            {
                std::string step( line.begin(), line.end() );
                trimStep( step );

                if ( !step.empty() )
                    steps.push_back( { std::move( step ), "pending" } );
            }

            if ( steps.empty() )
                throw std::runtime_error( "the plan has no steps; give one step a line" );

            if ( steps.size() > kMostSteps )
                throw std::runtime_error( std::format( "a plan has at most {} steps; join some", kMostSteps ) );

            steps_ = std::move( steps );

            return std::format( "The user approved the plan of {} steps. Work through them in order, and mark each with "
                "update_step as you start it and as you finish it.", steps_.size() );
        }

        /// `step`, from 1, and `status`: in_progress, done or skipped.
        std::string update( const std::string& arguments_text )
        {
            const auto arguments = nlohmann::json::parse( arguments_text );
            const std::int64_t number = arguments.at( "step" ).get<std::int64_t>();
            const std::string status = arguments.at( "status" ).get<std::string>();

            if ( steps_.empty() )
                throw std::runtime_error( "there is no plan yet; propose one with propose_plan" );

            if ( number < 1 || number > static_cast<std::int64_t>( steps_.size() ) )
                throw std::runtime_error( std::format( "the plan has steps 1 to {}, not {}", steps_.size(), number ) );

            if ( status != "in_progress" && status != "done" && status != "skipped" )
                throw std::runtime_error( std::format( "a step's status is in_progress, done or skipped, not '{}'", status ) );

            steps_[ static_cast<std::size_t>( number - 1 ) ].status = status;

            const auto finished = std::ranges::count_if( steps_, []( const Step& step ) { return step.status != "pending" && step.status != "in_progress"; } );

            return std::format( "Step {} is {}; {} of {} steps are finished.", number, status, finished, steps_.size() );
        }

        void clear() noexcept
        {
            steps_.clear();
        }

        /// The plan as `session.state` and the stored conversation carry it: [{text, status}], or null with none.
        [[nodiscard]] nlohmann::json toJson() const
        {
            if ( steps_.empty() )
                return nullptr;

            nlohmann::json steps = nlohmann::json::array();

            for ( const Step& step : steps_ )
                steps.push_back( { { "text", step.text }, { "status", step.status } } );

            return steps;
        }

        /// A stored plan, as toJson() wrote it; anything else is no plan.
        void restore( const nlohmann::json& stored )
        {
            steps_.clear();

            if ( !stored.is_array() )
                return;

            for ( const nlohmann::json& step : stored )
            {
                if ( step.is_object() && step.value( "text", nlohmann::json() ).is_string() )
                    steps_.push_back( { step[ "text" ].get<std::string>(), step.value( "status", std::string( "pending" ) ) } );
            }
        }

    private:

        static constexpr std::size_t kMostSteps = 20;

        struct Step
        {
            std::string text;
            std::string status;
        };

        /// A step as written in a list: its bullet or number and the spaces around it are not the step.
        static void trimStep( std::string& step )
        {
            std::size_t first = step.find_first_not_of( " \t\r" );

            if ( first == std::string::npos )
            {
                step.clear();

                return;
            }

            if ( step[ first ] == '-' || step[ first ] == '*' )
            {
                ++first;
            }
            else
            {
                const std::size_t digits = step.find_first_not_of( "0123456789", first );

                if ( digits != first && digits != std::string::npos && ( step[ digits ] == '.' || step[ digits ] == ')' ) )
                    first = digits + 1;
            }

            first = step.find_first_not_of( " \t", first );
            const auto last = step.find_last_not_of( " \t\r" );

            step = first == std::string::npos || first > last ? std::string() : step.substr( first, last - first + 1 );
        }

        std::vector<Step> steps_;
    };
}
